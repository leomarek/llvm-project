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
  if (Name.contains("LW") || Name.contains("LD") || Name.contains("LOAD"))
    return OpClass::Load;
  if (Name.contains("SW") || Name.contains("SD") || Name.contains("STORE"))
    return OpClass::Store;
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

static Register getMemoryBaseReg(const MachineInstr &MI) {
  for (int I = static_cast<int>(MI.getNumOperands()) - 1; I >= 0; --I) {
    const MachineOperand &MO = MI.getOperand(I);
    if (!MO.isReg() || !MO.isUse())
      continue;
    Register Reg = MO.getReg();
    if (Reg)
      return Reg;
  }
  return Register();
}

static bool isLikelyLSUOp(const MachineInstr &MI, const RISCVInstrInfo &TII) {
  if (MI.mayLoad() || MI.mayStore())
    return true;
  StringRef Name = TII.getName(MI.getOpcode());
  return Name.contains("LW") || Name.contains("LD") || Name.contains("LOAD") ||
         Name.contains("SW") || Name.contains("SD") || Name.contains("STORE");
}

static bool mayReorderAcross(const MachineInstr &A, const MachineInstr &B,
                             bool AssumeNoMemoryAlias) {
  // Conservative memory ordering model: do not move stores across memory ops,
  // and do not move loads above older stores unless we explicitly opt into
  // aggressive no-alias scheduling and can see disjoint base registers.
  if ((A.mayStore() && (B.mayLoad() || B.mayStore())) ||
      (A.mayLoad() && B.mayStore() && !AssumeNoMemoryAlias))
    return false;
  if (A.mayLoad() && B.mayStore() && AssumeNoMemoryAlias) {
    Register LoadBase = getMemoryBaseReg(A);
    Register StoreBase = getMemoryBaseReg(B);
    // Still conservative when base is unknown or matches.
    if (!LoadBase || !StoreBase || LoadBase == StoreBase)
      return false;
  }
  return true;
}

static bool canHoistCandidate(MachineBasicBlock::iterator InsertPos,
                              MachineBasicBlock::iterator CandIt,
                              bool PacketizePCRelative,
                              bool AssumeNoMemoryAlias) {
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

    if (!mayReorderAcross(Cand, Intervening, AssumeNoMemoryAlias))
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

  auto isLaneBusy = [&](ArrayRef<unsigned> UsedLanes, unsigned Lane) {
    for (unsigned UsedLane : UsedLanes)
      if (UsedLane == Lane)
        return true;
    return false;
  };

  auto tryAssignLaneWithUsed = [&](OpClass Class,
                                   ArrayRef<unsigned> UsedLanes) -> int {
    const unsigned NumLanes = Config.maxBundleWidth();

    // Prefer non-zero lanes to keep lane 0 available for unrestricted ops.
    if (Config.Scheduler.ReserveLane0ForAny && NumLanes > 1) {
      for (unsigned Lane = 1; Lane < NumLanes; ++Lane) {
        if (Config.isLaneLegal(Lane, Class) && !isLaneBusy(UsedLanes, Lane))
          return static_cast<int>(Lane);
      }
      if (Config.isLaneLegal(0, Class) && !isLaneBusy(UsedLanes, 0))
        return 0;
      return -1;
    }

    for (unsigned Lane = 0; Lane < NumLanes; ++Lane) {
      if (!Config.isLaneLegal(Lane, Class))
        continue;
      if (!isLaneBusy(UsedLanes, Lane))
        return static_cast<int>(Lane);
    }
    return -1;
  };

  auto tryAssignLane = [&](OpClass Class) -> int {
    return tryAssignLaneWithUsed(Class, PacketLanes);
  };

  auto countLaneChoices = [&](OpClass Class, ArrayRef<unsigned> UsedLanes) {
    const unsigned NumLanes = Config.maxBundleWidth();
    unsigned NumChoices = 0;
    for (unsigned Lane = 0; Lane < NumLanes; ++Lane) {
      if (!Config.isLaneLegal(Lane, Class) || isLaneBusy(UsedLanes, Lane))
        continue;
      ++NumChoices;
    }
    return NumChoices;
  };

  auto isLikelyPointerBump = [&](const MachineInstr &MI) {
    if (!Config.Scheduler.PrioritizePointerBumps)
      return false;
    const StringRef Name = TII->getName(MI.getOpcode());
    if (!Name.contains("ADDI"))
      return false;

    Register DefReg;
    Register UseReg;
    bool HasImm = false;
    int64_t Imm = 0;
    for (const MachineOperand &MO : MI.operands()) {
      if (MO.isReg() && MO.isDef() && !DefReg)
        DefReg = MO.getReg();
      if (MO.isReg() && MO.isUse() && !UseReg)
        UseReg = MO.getReg();
      if (MO.isImm()) {
        HasImm = true;
        Imm = MO.getImm();
      }
    }
    return DefReg && UseReg && DefReg == UseReg && HasImm && Imm != 0;
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
    unsigned MemoryOps = 0;
    int FirstMemoryIndex = -1;
    for (unsigned I = 0; I < Packet.size(); ++I) {
      if (!isLikelyLSUOp(*Packet[I], *TII))
        continue;
      ++MemoryOps;
      if (FirstMemoryIndex < 0)
        FirstMemoryIndex = static_cast<int>(I);
    }
    const bool IsLSULegalPacket =
        MemoryOps <= 1 && (FirstMemoryIndex < 0 || FirstMemoryIndex == 0);

    bool ShouldEmitHint = false;
    if (!HasDenseLanePrefix || !IsLSULegalPacket)
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

  auto estimateAdditionalFill = [&](MachineBasicBlock::iterator InsertPos,
                                    MachineBasicBlock::iterator ChosenIt,
                                    const DenseSet<Register> &ChosenUses,
                                    const DenseSet<Register> &ChosenDefs,
                                    unsigned ChosenLane) {
    if (Packet.size() + 1 >= Config.maxBundleWidth())
      return 0U;

    DenseSet<Register> SimUses(PacketUses);
    DenseSet<Register> SimDefs(PacketDefs);
    SimUses.insert(ChosenUses.begin(), ChosenUses.end());
    SimDefs.insert(ChosenDefs.begin(), ChosenDefs.end());

    SmallVector<unsigned, 8> SimLanes(PacketLanes.begin(), PacketLanes.end());
    SimLanes.push_back(ChosenLane);
    bool SimHasLSU = false;
    for (MachineInstr *PMI : Packet)
      SimHasLSU |= isLikelyLSUOp(*PMI, *TII);
    SimHasLSU |= isLikelyLSUOp(*ChosenIt, *TII);

    const unsigned LookAhead = std::max(1u, Config.Scheduler.PacketizerLookAhead);
    unsigned Seen = 0;
    unsigned Additional = 0;

    for (auto It = InsertPos; It != MBB.end() && Seen < LookAhead; ++It) {
      if (isBarrier(*It))
        break;
      ++Seen;
      if (It == ChosenIt)
        continue;

      if (!canHoistCandidate(InsertPos, It, Config.Scheduler.PacketizePCRelative,
                             Config.Scheduler.AssumeNoMemoryAlias))
        continue;

      DenseSet<Register> CandUses;
      DenseSet<Register> CandDefs;
      collectRegAccesses(*It, CandUses, CandDefs);

      if (hasPacketDependency(CandUses, CandDefs, SimUses, SimDefs))
        continue;

      const OpClass CandClass = classifyOpClass(*It, *TII);
      const int CandLane = tryAssignLaneWithUsed(CandClass, SimLanes);
      if (CandLane < 0)
        continue;
      const bool CandIsLSU = isLikelyLSUOp(*It, *TII);
      if (CandIsLSU && (SimHasLSU || CandLane != 0))
        continue;

      SimLanes.push_back(static_cast<unsigned>(CandLane));
      SimUses.insert(CandUses.begin(), CandUses.end());
      SimDefs.insert(CandDefs.begin(), CandDefs.end());
      SimHasLSU |= CandIsLSU;
      ++Additional;
      if (SimLanes.size() >= Config.maxBundleWidth())
        break;
    }

    return Additional;
  };

  auto hasRecentProducerUse = [&](MachineBasicBlock::iterator InsertPos,
                                  const DenseSet<Register> &CandUses,
                                  OpClass ProducerClass) {
    const unsigned Lookback =
        std::max(1u, Config.Scheduler.DependencyLookback);
    unsigned Seen = 0;
    for (auto It = InsertPos; It != MBB.begin() && Seen < Lookback;) {
      --It;
      if (isBarrier(*It))
        break;

      DenseSet<Register> PrevUses;
      DenseSet<Register> PrevDefs;
      collectRegAccesses(*It, PrevUses, PrevDefs);
      ++Seen;
      if (PrevDefs.empty())
        continue;
      if (classifyOpClass(*It, *TII) != ProducerClass)
        continue;
      for (Register Reg : PrevDefs) {
        if (CandUses.contains(Reg))
          return true;
      }
    }
    return false;
  };

  auto scoreCandidate = [&](const MachineInstr &MI, OpClass Class,
                            unsigned Lane, bool IsAtHead,
                            unsigned NumLaneChoices,
                            unsigned AdditionalFill, bool PacketHasLSU,
                            bool IsLSU,
                            MachineBasicBlock::iterator InsertPos,
                            const DenseSet<Register> &CandUses) {
    int Score = 0;
    // Bias toward candidates that allow this packet to become wider.
    Score += static_cast<int>(AdditionalFill) * 24;
    // Schedule constrained-lane ops earlier to avoid painting into a corner.
    Score += static_cast<int>(Config.maxBundleWidth() - NumLaneChoices) * 7;
    if (NumLaneChoices == 1)
      Score += 12;
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
    if (Class == OpClass::Load || Class == OpClass::Store)
      Score += 6;
    if (isLikelyPointerBump(MI))
      Score += 8;
    if (Config.Scheduler.PreferLSUAnchoredPackets) {
      if (Packet.empty()) {
        if (IsLSU)
          Score += 40;
        else
          Score -= 20;
      } else if (PacketHasLSU && !IsLSU) {
        // Prefer using non-LSU lanes while LSU lane is occupied by the anchor.
        if (Class == OpClass::ALUMulDiv)
          Score += 24;
        else if (Class == OpClass::ALUAddSub)
          Score += 20;
        else
          Score += 8;
      }
    }
    if (Config.Scheduler.PreferMACOverlap) {
      // Build an LSU->MUL->ADD->STORE stream when dependencies allow.
      if (Class == OpClass::ALUMulDiv &&
          hasRecentProducerUse(InsertPos, CandUses, OpClass::Load))
        Score += 30;
      if (Class == OpClass::ALUAddSub &&
          hasRecentProducerUse(InsertPos, CandUses, OpClass::ALUMulDiv))
        Score += 34;
      if (Class == OpClass::Store &&
          hasRecentProducerUse(InsertPos, CandUses, OpClass::ALUAddSub))
        Score += 18;
    }
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
    const bool PacketHasLSU =
        llvm::any_of(Packet, [&](const MachineInstr *PMI) {
          return isLikelyLSUOp(*PMI, *TII);
        });

    unsigned LookAhead = std::max(1u, Config.Scheduler.PacketizerLookAhead);
    unsigned Seen = 0;
    for (auto CandIt = MII; CandIt != MBB.end() && Seen < LookAhead; ++CandIt) {
      if (isBarrier(*CandIt))
        break;
      ++Seen;

      if (!canHoistCandidate(MII, CandIt, Config.Scheduler.PacketizePCRelative,
                             Config.Scheduler.AssumeNoMemoryAlias))
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
      const bool CandIsLSU = isLikelyLSUOp(*CandIt, *TII);
      if (CandIsLSU) {
        if (PacketHasLSU || CandLane != 0)
          continue;
      }
      const unsigned NumLaneChoices = countLaneChoices(CandClass, PacketLanes);
      const unsigned AdditionalFill =
          estimateAdditionalFill(MII, CandIt, CandUses, CandDefs,
                                 static_cast<unsigned>(CandLane));

      int CandScore = scoreCandidate(*CandIt, CandClass,
                                     static_cast<unsigned>(CandLane),
                                     CandIt == MII, NumLaneChoices,
                                     AdditionalFill, PacketHasLSU, CandIsLSU,
                                     MII, CandUses);
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
