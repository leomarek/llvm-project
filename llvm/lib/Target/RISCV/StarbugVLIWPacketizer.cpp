#include "StarbugVLIWPacketizer.h"

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Target/TargetMachine.h"

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

  auto tryAssignLane = [&](OpClass Class) -> int {
    const unsigned NumLanes = Config.maxBundleWidth();
    for (unsigned Lane = 0; Lane < NumLanes; ++Lane) {
      if (!Config.isLaneLegal(Lane, Class))
        continue;

      bool LaneBusy = false;
      for (unsigned UsedLane : PacketLanes) {
        if (UsedLane == Lane) {
          LaneBusy = true;
          break;
        }
      }
      if (!LaneBusy)
        return static_cast<int>(Lane);
    }
    return -1;
  };

  auto flushPacket = [&]() {
    if (Packet.empty())
      return;

    const size_t PacketSize = Packet.size();
    const bool IsSingle = PacketSize == 1;
    const bool IsFull = PacketSize == Config.maxBundleWidth();

    bool ShouldEmitHint = false;
    if (IsFull)
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

  for (MachineInstr &MI : MBB) {
    if (!isPacketizableMI(MI)) {
      flushPacket();
      continue;
    }

    DenseSet<Register> MIUses;
    DenseSet<Register> MIDefs;
    collectRegAccesses(MI, MIUses, MIDefs);

    if (!Packet.empty() &&
        hasPacketDependency(MIUses, MIDefs, PacketUses, PacketDefs)) {
      flushPacket();
    }

    OpClass Class = classifyOpClass(MI, *TII);
    int Lane = tryAssignLane(Class);
    if (Lane < 0) {
      flushPacket();
      Lane = tryAssignLane(Class);
    }

    if (Lane < 0)
      continue;

    Packet.push_back(&MI);
    PacketLanes.push_back(static_cast<unsigned>(Lane));
    PacketUses.insert(MIUses.begin(), MIUses.end());
    PacketDefs.insert(MIDefs.begin(), MIDefs.end());

    if (Packet.size() >= Config.maxBundleWidth())
      flushPacket();
  }

  flushPacket();

  return Changed;
}
