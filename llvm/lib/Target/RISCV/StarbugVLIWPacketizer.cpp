#include "StarbugVLIWPacketizer.h"

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>
#include <limits>
#include <numeric>

using namespace llvm;

#define DEBUG_TYPE "starbug-vliw-packetizer"
#define STARBUG_VLIW_PACKETIZER_NAME "Starbug VLIW Packetizer"

namespace {

using OpClass = RISCVVLIW::OpClass;

static bool isPacketizableMI(const MachineInstr &MI) {
  if (MI.isDebugInstr() || MI.isMetaInstruction() ||
      MI.getOpcode() == TargetOpcode::BUNDLE)
    return false;

  if (MI.isInlineAsm() || MI.isCall() || MI.isTerminator() || MI.isPseudo())
    return false;

  // Keep potentially stateful/ordered instructions serialized.
  if (MI.hasUnmodeledSideEffects())
    return false;

  return true;
}

static bool hasPCRelocationOperand(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.operands()) {
    const unsigned DirectTF =
        MO.getTargetFlags() & RISCVII::MO_DIRECT_FLAG_MASK;
    switch (DirectTF) {
    case RISCVII::MO_CALL:
    case RISCVII::MO_PCREL_LO:
    case RISCVII::MO_PCREL_HI:
    case RISCVII::MO_GOT_HI:
    case RISCVII::MO_TLS_GOT_HI:
    case RISCVII::MO_TLS_GD_HI:
    case RISCVII::MO_TLSDESC_HI:
    case RISCVII::MO_TLSDESC_LOAD_LO:
    case RISCVII::MO_TLSDESC_ADD_LO:
    case RISCVII::MO_TLSDESC_CALL:
      return true;
    default:
      break;
    }
  }
  return false;
}

static bool isPCRelativeSetupMI(const MachineInstr &MI) {
  if (MI.getOpcode() == RISCV::AUIPC)
    return true;
  return hasPCRelocationOperand(MI);
}

static void collectRegAccesses(const MachineInstr &MI, DenseSet<Register> &Uses,
                               DenseSet<Register> &Defs) {
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg())
      continue;

    Register Reg = MO.getReg();
    if (!Reg)
      continue;

    if (MO.isUse())
      Uses.insert(Reg);
    if (MO.isDef())
      Defs.insert(Reg);
  }
}

static bool hasPacketDependency(const DenseSet<Register> &MIUses,
                                const DenseSet<Register> &MIDefs,
                                const DenseSet<Register> &PacketUses,
                                const DenseSet<Register> &PacketDefs) {
  // RAW: current instruction reads what packet writes.
  for (Register Reg : MIUses)
    if (PacketDefs.contains(Reg))
      return true;

  // WAW/WAR: concurrent writes or write-after-read in one packet.
  for (Register Reg : MIDefs)
    if (PacketDefs.contains(Reg) || PacketUses.contains(Reg))
      return true;

  return false;
}

static OpClass classifyOpClass(const MachineInstr &MI,
                               const RISCVInstrInfo &TII) {
  if (MI.isBranch() || MI.isTerminator() || MI.isCall() || MI.isReturn())
    return OpClass::Branch;
  if (MI.mayLoad())
    return OpClass::Load;
  if (MI.mayStore())
    return OpClass::Store;

  StringRef Name = TII.getName(MI.getOpcode());
  if (Name.contains("ADD") || Name.contains("SUB"))
    return OpClass::ALUAddSub;
  if (Name.contains("MUL") || Name.contains("DIV") || Name.contains("REM"))
    return OpClass::ALUMulDiv;
  if (Name.contains("SLL") || Name.contains("SRL") || Name.contains("SRA") ||
      Name.contains("ROL") || Name.contains("ROR") || Name.contains("SH"))
    return OpClass::ALUShift;
  if (Name.contains("CSR"))
    return OpClass::CSR;
  if (Name.contains("MV") || Name.contains("LI") || Name.contains("COPY"))
    return OpClass::Move;
  if (Name.contains("SLT") || Name.contains("SEQ") || Name.contains("SNE") ||
      Name.contains("SGE") || Name.contains("SGT"))
    return OpClass::Compare;

  return OpClass::Any;
}

static bool mayReorderAcross(const MachineInstr &A, const MachineInstr &B) {
  // Conservative memory ordering model: do not move stores across memory ops,
  // and do not move loads above older stores.
  if ((A.mayStore() && (B.mayLoad() || B.mayStore())) ||
      (A.mayLoad() && B.mayStore()))
    return false;
  return true;
}

static bool canHoistCandidate(MachineBasicBlock::iterator InsertPos,
                              MachineBasicBlock::iterator CandIt,
                              bool PacketizePCRelative) {
  if (InsertPos == CandIt)
    return true;

  const MachineInstr &Cand = *CandIt;
  DenseSet<Register> CandUses;
  DenseSet<Register> CandDefs;
  collectRegAccesses(Cand, CandUses, CandDefs);

  for (auto It = InsertPos; It != CandIt; ++It) {
    const MachineInstr &Intervening = *It;
    if ((!PacketizePCRelative && isPCRelativeSetupMI(Intervening)) ||
        !isPacketizableMI(Intervening))
      return false;

    DenseSet<Register> IUses;
    DenseSet<Register> IDefs;
    collectRegAccesses(Intervening, IUses, IDefs);

    if (hasPacketDependency(CandUses, CandDefs, IUses, IDefs) ||
        hasPacketDependency(IUses, IDefs, CandUses, CandDefs))
      return false;

    if (!mayReorderAcross(Cand, Intervening))
      return false;
  }

  return true;
}

} // namespace

char StarbugVLIWPacketizer::ID = 0;

INITIALIZE_PASS(StarbugVLIWPacketizer, "starbug-vliw-packetizer",
                STARBUG_VLIW_PACKETIZER_NAME, false, false)

StarbugVLIWPacketizer::StarbugVLIWPacketizer()
    : MachineFunctionPass(ID),
      Config(RISCVVLIW::StarbugVLIWConfig::fromCommandLine()) {}

StringRef StarbugVLIWPacketizer::getPassName() const {
  return STARBUG_VLIW_PACKETIZER_NAME;
}

FunctionPass *llvm::createStarbugVLIWPacketizerPass() {
  return new StarbugVLIWPacketizer();
}

bool StarbugVLIWPacketizer::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;
  const auto &ST = MF.getSubtarget<RISCVSubtarget>();
  if (!ST.hasStarbugVLIW() || skipFunction(MF.getFunction()))
    return false;

  for (auto &MBB : MF)
    Changed |= packetizeBasicBlock(MBB, ST);

  return Changed;
}

bool StarbugVLIWPacketizer::packetizeBasicBlock(MachineBasicBlock &MBB,
                                                const RISCVSubtarget &ST) {
  const auto *TII = ST.getInstrInfo();
  bool Changed = false;
  if (!TII)
    return false;

  SmallVector<MachineInstr *, 8> Packet;
  SmallVector<unsigned, 8> PacketLanes;
  DenseSet<Register> PacketUses;
  DenseSet<Register> PacketDefs;

  auto isBarrier = [&](const MachineInstr &MI) {
    if (!Config.Scheduler.PacketizePCRelative && isPCRelativeSetupMI(MI))
      return true;
    return !isPacketizableMI(MI);
  };

  auto tryAssignLane = [&](OpClass Class) -> int {
    const unsigned NumLanes = Config.maxBundleWidth();
    auto isLaneBusy = [&](unsigned Lane) {
      for (unsigned UsedLane : PacketLanes)
        if (UsedLane == Lane)
          return true;
      return false;
    };

    // Prefer non-zero lanes to keep lane 0 available for unrestricted ops.
    if (Config.Scheduler.ReserveLane0ForAny && NumLanes > 1) {
      for (unsigned Lane = 1; Lane < NumLanes; ++Lane) {
        if (Config.isLaneLegal(Lane, Class) && !isLaneBusy(Lane))
          return static_cast<int>(Lane);
      }
      if (Config.isLaneLegal(0, Class) && !isLaneBusy(0))
        return 0;
      return -1;
    }

    for (unsigned Lane = 0; Lane < NumLanes; ++Lane) {
      if (!Config.isLaneLegal(Lane, Class))
        continue;
      if (!isLaneBusy(Lane))
        return static_cast<int>(Lane);
    }
    return -1;
  };

  auto flushPacket = [&]() {
    if (Packet.empty())
      return;

    // The emitted instruction order after STARBUG_BUNDLE_HINT defines lane
    // slots in hardware. Reorder packet members by chosen lane so lane 0 op
    // is emitted first, lane 1 second, etc.
    SmallVector<unsigned, 8> PacketOrder(Packet.size());
    std::iota(PacketOrder.begin(), PacketOrder.end(), 0);
    llvm::stable_sort(PacketOrder, [&](unsigned LHS, unsigned RHS) {
      return PacketLanes[LHS] < PacketLanes[RHS];
    });

    auto InsertPos = Packet.front()->getIterator();
    for (unsigned Idx : PacketOrder) {
      MachineInstr *MI = Packet[Idx];
      if (MI->getIterator() != InsertPos) {
        MBB.splice(InsertPos, &MBB, MI->getIterator());
        Changed = true;
      }
      ++InsertPos;
    }

    SmallVector<MachineInstr *, 8> SortedPacket;
    SmallVector<unsigned, 8> SortedPacketLanes;
    SortedPacket.reserve(Packet.size());
    SortedPacketLanes.reserve(PacketLanes.size());
    for (unsigned Idx : PacketOrder) {
      SortedPacket.push_back(Packet[Idx]);
      SortedPacketLanes.push_back(PacketLanes[Idx]);
    }
    Packet.swap(SortedPacket);
    PacketLanes.swap(SortedPacketLanes);

    const size_t PacketSize = Packet.size();
    const bool IsSingle = PacketSize == 1;
    const bool IsFull = PacketSize == Config.maxBundleWidth();
    bool HasDenseLanePrefix = true;
    for (unsigned Lane = 0; Lane < PacketLanes.size(); ++Lane) {
      if (PacketLanes[Lane] != Lane) {
        HasDenseLanePrefix = false;
        break;
      }
    }

    bool ShouldEmitHint = false;
    if (!HasDenseLanePrefix)
      ShouldEmitHint = false;
    else if (IsFull)
      ShouldEmitHint = true;
    else if (!IsSingle && Config.Scheduler.AllowShortPackets)
      ShouldEmitHint = true;
    else if (IsSingle && Config.Scheduler.EmitSingleInstructionHints)
      ShouldEmitHint = true;

    if (ShouldEmitHint) {
      MachineInstr &First = *Packet.front();
      BuildMI(MBB, First, First.getDebugLoc(),
              TII->get(RISCV::STARBUG_BUNDLE_HINT))
          .addImm(static_cast<int64_t>(PacketSize));
      Changed = true;
    }

    Packet.clear();
    PacketLanes.clear();
    PacketUses.clear();
    PacketDefs.clear();
  };

  auto scoreCandidate = [&](const MachineInstr &MI, OpClass Class,
                            unsigned Lane, bool IsAtHead) {
    int Score = 0;
    if (IsAtHead)
      Score += 3;
    if (Lane != 0)
      Score += 10;
    if (Class == OpClass::Load && Config.Scheduler.PrioritizeReadyLoads)
      Score += static_cast<int>(Config.Scheduler.LatencyWeightLoad) * 4;
    if (Class == OpClass::ALUMulDiv)
      Score += static_cast<int>(Config.Scheduler.LatencyWeightMulDiv) * 3;
    if (Class == OpClass::Store)
      Score += static_cast<int>(Config.Scheduler.LatencyWeightStore);
    if (Class == OpClass::Any)
      Score -= 6;
    if (MI.mayLoad())
      Score += 2;
    return Score;
  };

  for (auto MII = MBB.begin(); MII != MBB.end();) {
    if (isBarrier(*MII)) {
      flushPacket();
      ++MII;
      continue;
    }

    auto BestIt = MBB.end();
    DenseSet<Register> BestUses;
    DenseSet<Register> BestDefs;
    OpClass BestClass = OpClass::Any;
    int BestScore = std::numeric_limits<int>::min();

    unsigned LookAhead = std::max(1u, Config.Scheduler.PacketizerLookAhead);
    unsigned Seen = 0;
    for (auto CandIt = MII; CandIt != MBB.end() && Seen < LookAhead; ++CandIt) {
      if (isBarrier(*CandIt))
        break;
      ++Seen;

      if (!canHoistCandidate(MII, CandIt, Config.Scheduler.PacketizePCRelative))
        continue;

      DenseSet<Register> CandUses;
      DenseSet<Register> CandDefs;
      collectRegAccesses(*CandIt, CandUses, CandDefs);

      if (!Packet.empty() &&
          hasPacketDependency(CandUses, CandDefs, PacketUses, PacketDefs))
        continue;

      OpClass CandClass = classifyOpClass(*CandIt, *TII);
      int CandLane = tryAssignLane(CandClass);
      if (CandLane < 0)
        continue;

      int CandScore = scoreCandidate(*CandIt, CandClass,
                                     static_cast<unsigned>(CandLane),
                                     CandIt == MII);
      if (BestIt != MBB.end() && CandScore <= BestScore)
        continue;

      BestIt = CandIt;
      BestUses = std::move(CandUses);
      BestDefs = std::move(CandDefs);
      BestClass = CandClass;
      BestScore = CandScore;
    }

    if (BestIt == MBB.end()) {
      if (Packet.empty())
        ++MII; // No legal lane/candidate even for an empty packet.
      else
        flushPacket();
      continue;
    }

    if (BestIt != MII) {
      MBB.splice(MII, &MBB, BestIt);
      Changed = true;
    }

    MachineInstr &Scheduled = *MII;
    int Lane = tryAssignLane(BestClass);
    if (Lane < 0) {
      flushPacket();
      continue;
    }

    Packet.push_back(&Scheduled);
    PacketLanes.push_back(static_cast<unsigned>(Lane));
    PacketUses.insert(BestUses.begin(), BestUses.end());
    PacketDefs.insert(BestDefs.begin(), BestDefs.end());

    ++MII;

    if (Packet.size() >= Config.maxBundleWidth())
      flushPacket();
  }

  flushPacket();

  return Changed;
}
