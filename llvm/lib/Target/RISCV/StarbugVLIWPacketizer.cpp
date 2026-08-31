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
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/InitializePasses.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>
#include <limits>
#include <numeric>

using namespace llvm;

#define DEBUG_TYPE "starbug-vliw-packetizer"
#define STARBUG_VLIW_PACKETIZER_NAME "Starbug VLIW Packetizer"

STATISTIC(NumPacketsEmitted, "Number of STARBUG bundle hints emitted");
STATISTIC(NumPacketsRejected,
          "Number of candidate packets rejected by the safety verifier");

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

// Debug instructions carry no semantics and emit no bytes, so the packetizer
// has to step over them rather than react to them: building with -g must not
// change code generation.
//
// They used to be caught by isPacketizableMI's meta-instruction filter, which
// is also what the packetizer uses as its barrier test -- so every DBG_VALUE
// flushed the packet in flight. On a -gdwarf-2 build of the CMSIS DSP kernels
// (which is what the benchmark Makefiles use) that cost roughly three quarters
// of all bundling, and it cost it only on the compiler side: hand-written
// assembly carries its hints literally and never noticed.
static bool isTransparentMI(const MachineInstr &MI, bool Enabled = true) {
  return Enabled && MI.isDebugInstr();
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

static bool overlapsAnyRegister(Register Reg, const DenseSet<Register> &Regs,
                                const TargetRegisterInfo &TRI) {
  for (Register Other : Regs) {
    if (TRI.regsOverlap(Reg, Other))
      return true;
  }
  return false;
}

// RAW or WAW between a candidate and the packet built so far. Both make
// parallel issue impossible: all lanes read the register file in one cycle and
// write it back in one cycle, so a RAW consumer sees the stale value and a WAW
// pair has no defined winner.
static bool hasPacketHazard(const DenseSet<Register> &MIUses,
                            const DenseSet<Register> &MIDefs,
                            const DenseSet<Register> &PacketUses,
                            const DenseSet<Register> &PacketDefs,
                            const TargetRegisterInfo &TRI) {
  (void)PacketUses;
  // RAW: current instruction reads what packet writes.
  for (Register Reg : MIUses)
    if (overlapsAnyRegister(Reg, PacketDefs, TRI))
      return true;

  // WAW: two lanes writing one register in one cycle.
  for (Register Reg : MIDefs)
    if (overlapsAnyRegister(Reg, PacketDefs, TRI))
      return true;

  return false;
}

// WAR: the candidate overwrites something an existing packet member reads.
// Harmless for *parallel* issue -- every lane reads before any lane writes back
// -- but fatal the moment the emitted order changes, because the fetch unit can
// decline the bundle and run the same bytes sequentially. Kept separate so the
// packetizer can accept it exactly when it is not going to reorder.
static bool hasPacketAntiDependence(const DenseSet<Register> &MIDefs,
                                    const DenseSet<Register> &PacketUses,
                                    const TargetRegisterInfo &TRI) {
  for (Register Reg : MIDefs)
    if (overlapsAnyRegister(Reg, PacketUses, TRI))
      return true;
  return false;
}

static bool hasPacketDependency(const DenseSet<Register> &MIUses,
                                const DenseSet<Register> &MIDefs,
                                const DenseSet<Register> &PacketUses,
                                const DenseSet<Register> &PacketDefs,
                                const TargetRegisterInfo &TRI) {
  return hasPacketHazard(MIUses, MIDefs, PacketUses, PacketDefs, TRI) ||
         hasPacketAntiDependence(MIDefs, PacketUses, TRI);
}

// Does this instruction read or write the f-register file?
//
// This runs post-RA, so every register operand is already physical and its
// register class answers the question directly. Testing the register file is
// better than enumerating opcodes here: it covers F, D and Zfh in one rule and
// it cannot silently miss an instruction the way an opcode list can.
static bool touchesFPRegisters(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg())
      continue;
    Register Reg = MO.getReg();
    if (!Reg || !Reg.isPhysical())
      continue;
    if (RISCV::FPR32RegClass.contains(Reg) ||
        RISCV::FPR64RegClass.contains(Reg) ||
        RISCV::FPR16RegClass.contains(Reg))
      return true;
  }
  return false;
}

// fdiv and fsqrt drive FDivBusyE, which wallypipelinedcore.sv ORs across the
// lanes into a core-wide execute stall (hazard.sv:87). They are still legal in
// a worker lane -- they are just not free, so they get their own class and can
// be excluded from worker lanes with -starbug-vliw-lane-op-classes for sweeps.
static bool isFPDivSqrtOpcode(unsigned Opcode) {
  switch (Opcode) {
  case RISCV::FDIV_S:  case RISCV::FDIV_D:  case RISCV::FDIV_H:
  case RISCV::FSQRT_S: case RISCV::FSQRT_D: case RISCV::FSQRT_H:
    return true;
  default:
    return false;
  }
}

// Classify by opcode, not by substring of the instruction's name.
//
// The previous name-matching version misfiled a lot of common code: XOR, OR,
// AND and SLT matched no pattern at all and fell through to OpClass::Any,
// which is lane-0 only, so any kernel built from logic ops was pinned to a
// single lane. "SH" also matched store-halfword, and "ADD" matched AMOADD.
// Anything genuinely unrecognised still lands on Any and stays in lane 0,
// which is the safe direction.
static OpClass classifyOpClass(const MachineInstr &MI,
                               const RISCVInstrInfo &TII,
                               bool PacketizeFP) {
  (void)TII; // classification is now opcode-driven; no name lookup needed

  // These predicates come from the instruction description and outrank any
  // opcode guess.
  if (MI.isBranch() || MI.isTerminator() || MI.isCall() || MI.isReturn())
    return OpClass::Branch;
  if (MI.mayLoad())
    return OpClass::Load;
  if (MI.mayStore())
    return OpClass::Store;

  // Floating point belongs in the worker lanes. wallypipelinedcore.sv
  // instantiates fpu_1/fpu_2/fpu_3 beside the lane IEUs, backs them with
  // fregfile_widened (four write ports), and routes each lane's FP-to-integer
  // results -- FIntResM_n, FCvtIntResW_n, FIntDivResultW_n -- back into that
  // lane's own IEU writeback, so fmv.x.w and fcvt.w.s are as legal as fadd.s.
  // FP loads and stores are not covered here: mayLoad/mayStore above already
  // claimed them for lane 0 along with the rest of the memory traffic.
  if (touchesFPRegisters(MI))
    return !PacketizeFP ? OpClass::Any
                        : (isFPDivSqrtOpcode(MI.getOpcode())
                               ? OpClass::FPUDivSqrt
                               : OpClass::FPU);

  switch (MI.getOpcode()) {
  case RISCV::ADD:  case RISCV::ADDI:  case RISCV::SUB:
  case RISCV::ADDW: case RISCV::ADDIW: case RISCV::SUBW:
  case RISCV::C_ADD: case RISCV::C_ADDI: case RISCV::C_ADDI16SP:
  case RISCV::C_ADDI4SPN: case RISCV::C_SUB:
  case RISCV::C_ADDW: case RISCV::C_SUBW:
    return OpClass::ALUAddSub;

  case RISCV::MUL:  case RISCV::MULH: case RISCV::MULHU: case RISCV::MULHSU:
  case RISCV::MULW:
  case RISCV::DIV:  case RISCV::DIVU: case RISCV::DIVW:  case RISCV::DIVUW:
  case RISCV::REM:  case RISCV::REMU: case RISCV::REMW:  case RISCV::REMUW:
    return OpClass::ALUMulDiv;

  case RISCV::SLL:  case RISCV::SLLI: case RISCV::SRL: case RISCV::SRLI:
  case RISCV::SRA:  case RISCV::SRAI:
  case RISCV::SLLW: case RISCV::SLLIW: case RISCV::SRLW: case RISCV::SRLIW:
  case RISCV::SRAW: case RISCV::SRAIW:
  case RISCV::C_SLLI: case RISCV::C_SRLI: case RISCV::C_SRAI:
    return OpClass::ALUShift;

  case RISCV::XOR: case RISCV::XORI: case RISCV::OR: case RISCV::ORI:
  case RISCV::AND: case RISCV::ANDI:
  case RISCV::C_XOR: case RISCV::C_OR: case RISCV::C_AND: case RISCV::C_ANDI:
    return OpClass::ALULogic;

  case RISCV::SLT: case RISCV::SLTI: case RISCV::SLTU: case RISCV::SLTIU:
    return OpClass::Compare;

  case RISCV::LUI: case RISCV::C_LUI: case RISCV::C_LI: case RISCV::C_MV:
  case TargetOpcode::COPY:
    return OpClass::Move;

  case RISCV::CSRRW: case RISCV::CSRRS: case RISCV::CSRRC:
  case RISCV::CSRRWI: case RISCV::CSRRSI: case RISCV::CSRRCI:
    return OpClass::CSR;

  default:
    return OpClass::Any;
  }
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

// Does this instruction occupy the single shared LSU? mayLoad/mayStore come
// from the instruction description and are authoritative; the old substring
// fallback additionally caught unrelated opcodes whose names merely contained
// "LD" or "SW" and needlessly kept them out of worker lanes.
static bool isLikelyLSUOp(const MachineInstr &MI, const RISCVInstrInfo &TII) {
  (void)TII;
  return MI.mayLoad() || MI.mayStore();
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

static bool hasMemoryBaseDefHazard(const MachineInstr &MemMI,
                                   const MachineInstr &OtherMI,
                                   const TargetRegisterInfo &TRI) {
  if (!MemMI.mayLoad() && !MemMI.mayStore())
    return false;

  Register Base = getMemoryBaseReg(MemMI);
  if (!Base)
    return false;

  DenseSet<Register> OtherUses;
  DenseSet<Register> OtherDefs;
  collectRegAccesses(OtherMI, OtherUses, OtherDefs);
  return overlapsAnyRegister(Base, OtherDefs, TRI);
}

static bool canHoistCandidate(MachineBasicBlock::iterator InsertPos,
                              MachineBasicBlock::iterator CandIt,
                              bool PacketizePCRelative,
                              bool AssumeNoMemoryAlias,
                              bool DebugTransparent,
                              const TargetRegisterInfo &TRI) {
  if (InsertPos == CandIt)
    return true;

  const MachineInstr &Cand = *CandIt;
  DenseSet<Register> CandUses;
  DenseSet<Register> CandDefs;
  collectRegAccesses(Cand, CandUses, CandDefs);

  for (auto It = InsertPos; It != CandIt; ++It) {
    const MachineInstr &Intervening = *It;
    if (isTransparentMI(Intervening, DebugTransparent))
      continue;
    if ((!PacketizePCRelative && isPCRelativeSetupMI(Intervening)) ||
        !isPacketizableMI(Intervening))
      return false;

    DenseSet<Register> IUses;
    DenseSet<Register> IDefs;
    collectRegAccesses(Intervening, IUses, IDefs);

    if (hasPacketDependency(CandUses, CandDefs, IUses, IDefs, TRI) ||
        hasPacketDependency(IUses, IDefs, CandUses, CandDefs, TRI))
      return false;

    if (hasMemoryBaseDefHazard(Cand, Intervening, TRI) ||
        hasMemoryBaseDefHazard(Intervening, Cand, TRI))
      return false;

    if (!mayReorderAcross(Cand, Intervening, AssumeNoMemoryAlias))
      return false;
  }

  return true;
}

// Instructions that only lane 0 can execute. Worker lanes have their LSU
// address path (IEUAdrE) and branch path (PCSrcE) disconnected in the STARBUG
// RTL, and their MemRW outputs are not wired to the LSU at all -- a memory op
// placed in a worker lane is silently dropped rather than faulting. Anything
// reading the PC is also lane-0 only because every lane is fed the same PCE.
static bool isLane0OnlyMI(const MachineInstr &MI) {
  return MI.mayLoad() || MI.mayStore() || MI.isBranch() || MI.isCall() ||
         MI.isReturn() || MI.isTerminator() || MI.isBarrier() ||
         MI.hasUnmodeledSideEffects() || isPCRelativeSetupMI(MI);
}

/// Are these instructions independent enough to be freely permuted?
///
/// The packetizer reorders packet members so that lane assignment matches
/// emission order. That permutation has to be correct under *sequential*
/// execution as well as parallel, because the fetch unit silently declines a
/// bundle whenever it straddles an I-cache line, is uncacheable, or comes from
/// the IROM -- and then runs the very same bytes scalar, in memory order. The
/// whole ISA-compatibility story rests on that fallback being correct.
///
/// So permutation requires full independence: RAW and WAW (which parallel
/// issue also requires) plus WAR (which only matters once the order changes).
static bool arePacketMembersIndependent(ArrayRef<MachineInstr *> Members,
                                        const TargetRegisterInfo &TRI) {
  for (unsigned I = 0; I < Members.size(); ++I) {
    DenseSet<Register> IUses, IDefs;
    collectRegAccesses(*Members[I], IUses, IDefs);
    for (unsigned J = I + 1; J < Members.size(); ++J) {
      DenseSet<Register> JUses, JDefs;
      collectRegAccesses(*Members[J], JUses, JDefs);
      if (hasPacketDependency(JUses, JDefs, IUses, IDefs, TRI))
        return false;
    }
  }
  return true;
}

/// Final safety gate for a formed packet.
///
/// STARBUG hardware trusts the HINT unconditionally: there is no interlock
/// that detects a bundle whose members are actually dependent, so a bad hint
/// is a silent wrong answer rather than a fault. This re-derives the facts
/// from the *final* instruction sequence that will be emitted, independently
/// of the scheduling heuristics that produced it. If anything fails we simply
/// decline to emit the hint; the same instructions then execute scalar, which
/// is always correct.
static bool isPacketSafeToHint(MachineBasicBlock &MBB, MachineInstr &First,
                               unsigned PacketSize, bool DebugTransparent,
                               const TargetRegisterInfo &TRI) {
  if (PacketSize == 0)
    return false;

  // Collect the instructions that will actually follow the hint, in program
  // order, rather than trusting the packet bookkeeping.
  SmallVector<MachineInstr *, 8> Members;
  auto It = First.getIterator();
  for (unsigned I = 0; I < PacketSize; ++I) {
    // Debug instructions emit nothing, so they do not sit between the hint and
    // its members in the encoded stream and must not be counted here.
    while (It != MBB.end() && isTransparentMI(*It, DebugTransparent))
      ++It;
    if (It == MBB.end())
      return false;
    // Anything else that is not a real, packetizable instruction breaks the
    // "next N instructions" contract the hint encodes.
    if (!isPacketizableMI(*It))
      return false;
    Members.push_back(&*It);
    ++It;
  }

  unsigned MemoryOps = 0;
  for (unsigned I = 0; I < Members.size(); ++I) {
    const MachineInstr &MI = *Members[I];
    const bool IsMem = MI.mayLoad() || MI.mayStore();
    if (IsMem) {
      ++MemoryOps;
      // The single LSU is wired to lane 0.
      if (I != 0)
        return false;
    }
    if (I != 0 && isLane0OnlyMI(MI))
      return false;
  }
  if (MemoryOps > 1)
    return false;

  // Every lane reads the register file in the same cycle and writes back in
  // the same cycle, so a RAW pair can never be satisfied and a WAW pair has no
  // defined winner. Both must be absent. (WAR is architecturally safe here and
  // is deliberately permitted.)
  for (unsigned I = 0; I < Members.size(); ++I) {
    DenseSet<Register> IUses, IDefs;
    collectRegAccesses(*Members[I], IUses, IDefs);
    for (unsigned J = I + 1; J < Members.size(); ++J) {
      DenseSet<Register> JUses, JDefs;
      collectRegAccesses(*Members[J], JUses, JDefs);

      for (Register Reg : JUses)               // RAW: J reads what I writes
        if (overlapsAnyRegister(Reg, IDefs, TRI))
          return false;
      for (Register Reg : JDefs)               // WAW: both write
        if (overlapsAnyRegister(Reg, IDefs, TRI))
          return false;
    }
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
  const auto *TRI = ST.getRegisterInfo();
  bool Changed = false;
  if (!TII || !TRI)
    return false;

  SmallVector<MachineInstr *, 8> Packet;
  SmallVector<unsigned, 8> PacketLanes;
  DenseSet<Register> PacketUses;
  DenseSet<Register> PacketDefs;

  // WAR is safe inside a bundle but not across a reordering (see
  // hasPacketAntiDependence). These two flags are what let the packetizer keep
  // an anti-dependent member: it accepts one only while the packet is still
  // exactly the original instruction sequence, and then refuses to permute.
  bool PacketInProgramOrder = true;
  bool PacketHasAntiDep = false;

  auto isBarrier = [&](const MachineInstr &MI) {
    if (isTransparentMI(MI, Config.Scheduler.DebugInstrsTransparent))
      return false;
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

  auto abandonPacket = [&]() {
    Packet.clear();
    PacketLanes.clear();
    PacketUses.clear();
    PacketDefs.clear();
    PacketInProgramOrder = true;
    PacketHasAntiDep = false;
  };

  auto flushPacket = [&]() {
    if (Packet.empty())
      return;

    // Check before touching the block, not after. If the members are not
    // mutually independent we must not permute them at all: the reordered
    // sequence would still be what executes when the fetch unit declines the
    // bundle. Leaving them in program order and emitting no hint costs
    // parallelism and nothing else.
    // A packet carrying an anti-dependence is deliberately exempt: it was only
    // allowed to form while it stayed in original program order, and the
    // ordering step below refuses to move anything in it. Everything else is
    // free to be permuted and must prove it.
    if (Packet.size() > 1 && !PacketHasAntiDep &&
        !arePacketMembersIndependent(Packet, *TRI)) {
      ++NumPacketsRejected;
      LLVM_DEBUG(dbgs() << "starbug: abandoning non-independent packet of size "
                        << Packet.size() << " in " << MBB.getName() << '\n');
      abandonPacket();
      return;
    }

    // Hardware assigns lanes by *position*: the instruction right after the
    // hint is lane 0, the next is lane 1, and so on, with no gaps possible.
    // The lane numbers chosen during packet building are only a reservation
    // scheme, and sorting by them used to leave lane 0 empty whenever every
    // member happened to be worker-legal -- which then failed the dense-prefix
    // test and threw the whole bundle away.
    //
    // Build the emission order directly instead: the one member that must own
    // lane 0 (a memory op, or anything else worker lanes cannot execute) goes
    // first, and the rest follow in program order. Positions are dense by
    // construction.
    SmallVector<unsigned, 8> PacketOrder;
    PacketOrder.reserve(Packet.size());
    int MustBeFirst = -1;
    for (unsigned I = 0; I < Packet.size(); ++I) {
      if (!isLikelyLSUOp(*Packet[I], *TII) && !isLane0OnlyMI(*Packet[I]))
        continue;
      if (MustBeFirst >= 0) {
        // Two members both need lane 0; no legal placement exists. Leave them
        // untouched and emit nothing.
        ++NumPacketsRejected;
        abandonPacket();
        return;
      }
      MustBeFirst = static_cast<int>(I);
    }
    // The one permutation this routine can still perform is hoisting the
    // lane-0-only member to the front. With an anti-dependence present that is
    // exactly the move that would break the scalar-fallback reading of these
    // bytes, so give the bundle up instead.
    if (MustBeFirst > 0 && PacketHasAntiDep) {
      ++NumPacketsRejected;
      LLVM_DEBUG(dbgs() << "starbug: anti-dependent packet needs a lane-0 "
                           "hoist; leaving it scalar in "
                        << MBB.getName() << '\n');
      abandonPacket();
      return;
    }

    if (MustBeFirst >= 0)
      PacketOrder.push_back(static_cast<unsigned>(MustBeFirst));
    for (unsigned I = 0; I < Packet.size(); ++I)
      if (static_cast<int>(I) != MustBeFirst)
        PacketOrder.push_back(I);

    // Rebuild the region in lane order. InsertPos stays anchored on the first
    // not-yet-placed instruction: splicing a member in front of it fills the
    // slot immediately before InsertPos, so InsertPos must only advance when
    // the instruction already sitting there is the one we wanted next.
    // (Advancing after a splice would step over an unplaced instruction and
    // silently drop it into, or out of, the bundle.)
    auto InsertPos = Packet.front()->getIterator();
    for (unsigned Idx : PacketOrder) {
      MachineInstr *MI = Packet[Idx];
      while (InsertPos != MBB.end() &&
             isTransparentMI(*InsertPos, Config.Scheduler.DebugInstrsTransparent))
        ++InsertPos;
      if (&*InsertPos == MI) {
        ++InsertPos;
        continue;
      }
      MBB.splice(InsertPos, &MBB, MI->getIterator());
      Changed = true;
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

    // Positions are dense by construction now, and the ordering step above
    // already guaranteed at most one lane-0-only member sitting at index 0.
    // isPacketSafeToHint re-derives both facts from the emitted sequence
    // below, so this only decides whether a bundle of this *size* is worth a
    // hint at all.
    bool ShouldEmitHint = false;
    if (IsFull)
      ShouldEmitHint = true;
    else if (!IsSingle && Config.Scheduler.AllowShortPackets)
      ShouldEmitHint = true;
    else if (IsSingle && Config.Scheduler.EmitSingleInstructionHints)
      ShouldEmitHint = true;

    // A hint is only ever a performance hint: declining to emit one costs
    // parallelism but never correctness. Verify the real emitted sequence and
    // stay scalar if it does not check out.
    if (ShouldEmitHint && PacketSize <= Config.maxBundleWidth()) {
      MachineInstr &First = *Packet.front();
      if (isPacketSafeToHint(MBB, First, PacketSize,
                             Config.Scheduler.DebugInstrsTransparent, *TRI)) {
        BuildMI(MBB, First, First.getDebugLoc(),
                TII->get(RISCV::STARBUG_BUNDLE_HINT))
            .addImm(static_cast<int64_t>(PacketSize));
        ++NumPacketsEmitted;
        Changed = true;
      } else {
        ++NumPacketsRejected;
        LLVM_DEBUG(dbgs() << "starbug: rejected unsafe packet of size "
                          << PacketSize << " in " << MBB.getName() << '\n');
      }
    }

    abandonPacket();
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
      if (isTransparentMI(*It, Config.Scheduler.DebugInstrsTransparent))
        continue;
      ++Seen;
      if (It == ChosenIt)
        continue;

      if (!canHoistCandidate(InsertPos, It,
                             Config.Scheduler.PacketizePCRelative,
                             Config.Scheduler.AssumeNoMemoryAlias,
                             Config.Scheduler.DebugInstrsTransparent, *TRI))
        continue;

      DenseSet<Register> CandUses;
      DenseSet<Register> CandDefs;
      collectRegAccesses(*It, CandUses, CandDefs);

      if (hasPacketDependency(CandUses, CandDefs, SimUses, SimDefs, *TRI))
        continue;

      bool BaseHazard = false;
      for (MachineInstr *PMI : Packet) {
        if (hasMemoryBaseDefHazard(*PMI, *It, *TRI) ||
            hasMemoryBaseDefHazard(*It, *PMI, *TRI)) {
          BaseHazard = true;
          break;
        }
      }
      if (BaseHazard)
        continue;

      const OpClass CandClass =
          classifyOpClass(*It, *TII, Config.Scheduler.PacketizeFP);
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
      if (isTransparentMI(*It, Config.Scheduler.DebugInstrsTransparent))
        continue;

      DenseSet<Register> PrevUses;
      DenseSet<Register> PrevDefs;
      collectRegAccesses(*It, PrevUses, PrevDefs);
      ++Seen;
      if (PrevDefs.empty())
        continue;
      if (classifyOpClass(*It, *TII, Config.Scheduler.PacketizeFP) !=
          ProducerClass)
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
    if (isTransparentMI(*MII, Config.Scheduler.DebugInstrsTransparent)) {
      // Emits no bytes: step over it and leave the packet in flight. Members
      // separated by one of these are still contiguous once encoded.
      ++MII;
      continue;
    }
    if (isBarrier(*MII)) {
      flushPacket();
      ++MII;
      continue;
    }

    auto BestIt = MBB.end();
    DenseSet<Register> BestUses;
    DenseSet<Register> BestDefs;
    OpClass BestClass = OpClass::Any;
    bool BestAntiDep = false;
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
      if (isTransparentMI(*CandIt, Config.Scheduler.DebugInstrsTransparent))
        continue;
      ++Seen;

      if (!canHoistCandidate(MII, CandIt,
                             Config.Scheduler.PacketizePCRelative,
                             Config.Scheduler.AssumeNoMemoryAlias,
                             Config.Scheduler.DebugInstrsTransparent, *TRI))
        continue;

      DenseSet<Register> CandUses;
      DenseSet<Register> CandDefs;
      collectRegAccesses(*CandIt, CandUses, CandDefs);

      bool CandAntiDep = false;
      if (!Packet.empty()) {
        if (hasPacketHazard(CandUses, CandDefs, PacketUses, PacketDefs, *TRI))
          continue;
        CandAntiDep = hasPacketAntiDependence(CandDefs, PacketUses, *TRI);
        // An anti-dependent member may join only if taking it changes nothing
        // about the order: it has to be at the head already, and every earlier
        // member has to have come from the head too. Then the bundle's bytes
        // are still the original program, which is what executes if the fetch
        // unit declines the bundle or a branch lands in the middle of it.
        if (CandAntiDep &&
            !(Config.Scheduler.AllowIntraPacketWAR && PacketInProgramOrder &&
              CandIt == MII))
          continue;
      }

      bool BaseHazard = false;
      for (MachineInstr *PMI : Packet) {
        if (hasMemoryBaseDefHazard(*PMI, *CandIt, *TRI) ||
            hasMemoryBaseDefHazard(*CandIt, *PMI, *TRI)) {
          BaseHazard = true;
          break;
        }
      }
      if (BaseHazard)
        continue;

      OpClass CandClass =
          classifyOpClass(*CandIt, *TII, Config.Scheduler.PacketizeFP);
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
      BestAntiDep = CandAntiDep;
      BestScore = CandScore;
    }

    if (BestIt == MBB.end()) {
      if (Packet.empty())
        ++MII; // No legal lane/candidate even for an empty packet.
      else
        flushPacket();
      continue;
    }

    // Invariant: every packet member sits contiguously immediately before MII.
    //
    // MBB.splice(MII, ...) inserts the candidate *before* the instruction MII
    // denotes, and leaves MII pointing at that same (still unscheduled)
    // instruction. So the instruction we just scheduled is std::prev(MII), and
    // MII must not advance -- it still refers to work we have not looked at.
    // Only when the winning candidate was already at the head do we consume it
    // and step forward.
    MachineBasicBlock::iterator ScheduledIt;
    if (BestIt != MII) {
      MBB.splice(MII, &MBB, BestIt);
      Changed = true;
      ScheduledIt = std::prev(MII);
      PacketInProgramOrder = false;
    } else {
      ScheduledIt = MII;
      ++MII;
    }

    MachineInstr &Scheduled = *ScheduledIt;
    int Lane = tryAssignLane(BestClass);
    if (Lane < 0) {
      flushPacket();
      continue;
    }

    Packet.push_back(&Scheduled);
    PacketLanes.push_back(static_cast<unsigned>(Lane));
    PacketHasAntiDep |= BestAntiDep;
    PacketUses.insert(BestUses.begin(), BestUses.end());
    PacketDefs.insert(BestDefs.begin(), BestDefs.end());

    if (Packet.size() >= Config.maxBundleWidth())
      flushPacket();
  }

  flushPacket();

  return Changed;
}
