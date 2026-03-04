#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "StarbugVLIWConfig.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "starbug-vliw-trace-scheduler"
#define STARBUG_VLIW_TRACE_SCHEDULER_NAME "Starbug VLIW Trace Scheduler"

namespace {

static cl::opt<unsigned> StarbugVLIWTraceMinUnits(
    "starbug-vliw-trace-min-units", cl::Hidden, cl::init(4),
    cl::desc("Minimum repeated load/load/arith/arith/store units for staging"));

static cl::opt<unsigned> StarbugVLIWTraceMaxUnits(
    "starbug-vliw-trace-max-units", cl::Hidden, cl::init(32),
    cl::desc("Maximum units to stage in one transformed trace"));

static cl::opt<unsigned> StarbugVLIWTraceMaxDefVRegs(
    "starbug-vliw-trace-max-def-vregs", cl::Hidden, cl::init(128),
    cl::desc("Skip trace staging when virtual def count exceeds this limit"));

struct TraceUnit {
  MachineInstr *Load0 = nullptr;
  MachineInstr *Load1 = nullptr;
  MachineInstr *Arith0 = nullptr;
  MachineInstr *Arith1 = nullptr;
  MachineInstr *Store = nullptr;

  DenseSet<Register> Load0Defs;
  DenseSet<Register> Load1Defs;
  DenseSet<Register> Arith0Defs;
  DenseSet<Register> Arith0Uses;
  DenseSet<Register> Arith1Defs;
  DenseSet<Register> Arith1Uses;

  Register Load0Def = Register();
  Register Load1Def = Register();
  Register Arith0Def = Register();
  Register Arith1Def = Register();
  Register StoreValue = Register();

  Register Load0Base;
  Register Load1Base;
  Register StoreBase;

  int64_t Load0Off = 0;
  int64_t Load1Off = 0;
  int64_t StoreOff = 0;
};

static bool isSimpleMI(const MachineInstr &MI) {
  if (MI.isDebugInstr() || MI.isMetaInstruction() || MI.isPseudo() ||
      MI.isInlineAsm() || MI.isCall() || MI.isTerminator())
    return false;
  if (MI.hasUnmodeledSideEffects())
    return false;
  return true;
}

static bool isLoadMI(const MachineInstr &MI) {
  return isSimpleMI(MI) && MI.mayLoad() && !MI.mayStore();
}

static bool isStoreMI(const MachineInstr &MI) {
  return isSimpleMI(MI) && MI.mayStore();
}

static bool isArithMI(const MachineInstr &MI, const RISCVInstrInfo &TII) {
  if (!isSimpleMI(MI) || MI.mayLoad() || MI.mayStore())
    return false;

  const StringRef Name = TII.getName(MI.getOpcode());
  if (Name.contains("ADDI") || Name.contains("SLTI") || Name.contains("ANDI") ||
      Name.contains("ORI") || Name.contains("XORI"))
    return false;

  return Name.contains("ADD") || Name.contains("SUB") || Name.contains("MUL") ||
         Name.contains("DIV") || Name.contains("REM") || Name.contains("SLL") ||
         Name.contains("SRL") || Name.contains("SRA") || Name.contains("XOR") ||
         Name.contains("AND") || Name.contains("OR") || Name.contains("MIN") ||
         Name.contains("MAX");
}

static void collectRegs(const MachineInstr &MI, DenseSet<Register> &Uses,
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

static Register extractMemoryBase(const MachineInstr &MI) {
  for (int I = static_cast<int>(MI.getNumOperands()) - 1; I >= 0; --I) {
    const MachineOperand &MO = MI.getOperand(I);
    if (MO.isReg() && MO.isUse() && MO.getReg())
      return MO.getReg();
  }
  return Register();
}

static Register extractStoreValue(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isReg() && MO.isUse() && MO.getReg())
      return MO.getReg();
  }
  return Register();
}

static bool extractImmOffset(const MachineInstr &MI, int64_t &Imm) {
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isImm()) {
      Imm = MO.getImm();
      return true;
    }
  }
  return false;
}

class StarbugVLIWTraceScheduler : public MachineFunctionPass {
public:
  static char ID;
  StarbugVLIWTraceScheduler()
      : MachineFunctionPass(ID),
        Config(RISCVVLIW::StarbugVLIWConfig::fromCommandLine()) {}

  StringRef getPassName() const override {
    return STARBUG_VLIW_TRACE_SCHEDULER_NAME;
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    const auto &ST = MF.getSubtarget<RISCVSubtarget>();
    if (!ST.hasStarbugVLIW() || !Config.Scheduler.AssumeNoMemoryAlias ||
        skipFunction(MF.getFunction()))
      return false;

    bool Changed = false;
    for (MachineBasicBlock &MBB : MF)
      Changed |= stageTraceInBlock(MBB, *ST.getInstrInfo(), MF.getRegInfo());
    return Changed;
  }

private:
  RISCVVLIW::StarbugVLIWConfig Config;

  static bool isConstStride(ArrayRef<int64_t> Offsets) {
    if (Offsets.size() <= 2)
      return true;
    const int64_t Stride = Offsets[1] - Offsets[0];
    if (Stride == 0)
      return false;
    for (size_t I = 2; I < Offsets.size(); ++I)
      if (Offsets[I] - Offsets[I - 1] != Stride)
        return false;
    return true;
  }

  static bool countDefVRegsWithinLimit(ArrayRef<TraceUnit> Units,
                                       const MachineRegisterInfo &MRI) {
    DenseSet<Register> Defs;
    for (const TraceUnit &U : Units) {
      for (Register R : U.Load0Defs)
        if (R.isVirtual())
          Defs.insert(R);
      for (Register R : U.Load1Defs)
        if (R.isVirtual())
          Defs.insert(R);
      for (Register R : U.Arith0Defs)
        if (R.isVirtual())
          Defs.insert(R);
      for (Register R : U.Arith1Defs)
        if (R.isVirtual())
          Defs.insert(R);
    }
    (void)MRI;
    return Defs.size() <= StarbugVLIWTraceMaxDefVRegs;
  }

  static bool matchUnit(MachineBasicBlock::iterator Begin,
                        MachineBasicBlock::iterator End, const RISCVInstrInfo &TII,
                        TraceUnit &Out, MachineBasicBlock::iterator &NextIt) {
    if (Begin == End)
      return false;

    auto It = Begin;
    if (!isLoadMI(*It))
      return false;
    Out.Load0 = &*It++;
    if (It == End || !isLoadMI(*It))
      return false;
    Out.Load1 = &*It++;
    if (It == End || !isArithMI(*It, TII))
      return false;
    Out.Arith0 = &*It++;
    if (It == End || !isArithMI(*It, TII))
      return false;
    Out.Arith1 = &*It++;
    if (It == End || !isStoreMI(*It))
      return false;
    Out.Store = &*It++;
    NextIt = It;

    DenseSet<Register> TmpUses;
    collectRegs(*Out.Load0, TmpUses, Out.Load0Defs);
    TmpUses.clear();
    collectRegs(*Out.Load1, TmpUses, Out.Load1Defs);
    collectRegs(*Out.Arith0, Out.Arith0Uses, Out.Arith0Defs);
    collectRegs(*Out.Arith1, Out.Arith1Uses, Out.Arith1Defs);

    if (Out.Load0Defs.size() != 1 || Out.Load1Defs.size() != 1 ||
        Out.Arith0Defs.size() != 1 || Out.Arith1Defs.size() != 1)
      return false;
    Out.Load0Def = *Out.Load0Defs.begin();
    Out.Load1Def = *Out.Load1Defs.begin();
    Out.Arith0Def = *Out.Arith0Defs.begin();
    Out.Arith1Def = *Out.Arith1Defs.begin();

    if (!Out.Arith0Uses.contains(Out.Load0Def) ||
        !Out.Arith0Uses.contains(Out.Load1Def))
      return false;
    if (!Out.Arith1Uses.contains(Out.Arith0Def))
      return false;

    Out.Load0Base = extractMemoryBase(*Out.Load0);
    Out.Load1Base = extractMemoryBase(*Out.Load1);
    Out.StoreBase = extractMemoryBase(*Out.Store);
    Out.StoreValue = extractStoreValue(*Out.Store);
    if (!Out.Load0Base || !Out.Load1Base || !Out.StoreBase || !Out.StoreValue)
      return false;
    if (Out.StoreValue == Out.StoreBase)
      return false;
    if (Out.StoreValue != Out.Arith1Def)
      return false;

    if (!extractImmOffset(*Out.Load0, Out.Load0Off) ||
        !extractImmOffset(*Out.Load1, Out.Load1Off) ||
        !extractImmOffset(*Out.Store, Out.StoreOff))
      return false;

    return true;
  }

  static bool allUsesWithin(const MachineRegisterInfo &MRI, Register DefReg,
                            ArrayRef<const MachineInstr *> AllowedUsers) {
    DenseSet<const MachineInstr *> Allowed(AllowedUsers.begin(),
                                           AllowedUsers.end());
    for (MachineOperand &MO : MRI.use_nodbg_operands(DefReg)) {
      const MachineInstr *User = MO.getParent();
      if (!Allowed.contains(User))
        return false;
    }
    return true;
  }

  bool verifyAndStage(ArrayRef<TraceUnit> Units, MachineBasicBlock &MBB,
                      const MachineRegisterInfo &MRI) const {
    if (Units.size() < StarbugVLIWTraceMinUnits)
      return false;

    const Register L0Base = Units.front().Load0Base;
    const Register L1Base = Units.front().Load1Base;
    const Register SBase = Units.front().StoreBase;
    if (SBase == L0Base || SBase == L1Base)
      return false;

    SmallVector<int64_t, 16> L0Offs;
    SmallVector<int64_t, 16> L1Offs;
    SmallVector<int64_t, 16> SOffs;
    L0Offs.reserve(Units.size());
    L1Offs.reserve(Units.size());
    SOffs.reserve(Units.size());

    for (size_t I = 0; I < Units.size(); ++I) {
      const TraceUnit &U = Units[I];
      if (U.Load0Base != L0Base || U.Load1Base != L1Base || U.StoreBase != SBase)
        return false;
      if (I > 0 && !U.Arith1Uses.contains(Units[I - 1].Arith1Def))
        return false;
      L0Offs.push_back(U.Load0Off);
      L1Offs.push_back(U.Load1Off);
      SOffs.push_back(U.StoreOff);
    }

    if (!isConstStride(L0Offs) || !isConstStride(L1Offs) || !isConstStride(SOffs))
      return false;

    // Keep the transform closed: every staged def must be consumed only by
    // the unit-local user(s) we model.
    for (size_t I = 0; I < Units.size(); ++I) {
      const TraceUnit &U = Units[I];
      if (!allUsesWithin(MRI, U.Load0Def, {U.Arith0}) ||
          !allUsesWithin(MRI, U.Load1Def, {U.Arith0}) ||
          !allUsesWithin(MRI, U.Arith0Def, {U.Arith1}))
        return false;

      if (I + 1 < Units.size()) {
        if (!allUsesWithin(MRI, U.Arith1Def, {U.Store, Units[I + 1].Arith1}))
          return false;
      } else if (!allUsesWithin(MRI, U.Arith1Def, {U.Store})) {
        return false;
      }
    }

    // Require the matched trace to be fully contiguous with no interleaving
    // instructions. This avoids moving unrelated epilogue/prologue code.
    DenseSet<const MachineInstr *> UnitInstrs;
    UnitInstrs.reserve(Units.size() * 5);
    for (const TraceUnit &U : Units) {
      UnitInstrs.insert(U.Load0);
      UnitInstrs.insert(U.Load1);
      UnitInstrs.insert(U.Arith0);
      UnitInstrs.insert(U.Arith1);
      UnitInstrs.insert(U.Store);
    }

    auto FirstIt = Units.front().Load0->getIterator();
    auto LastIt = Units.back().Store->getIterator();
    for (auto It = FirstIt;; ++It) {
      if (!UnitInstrs.contains(&*It))
        return false;
      if (It == LastIt)
        break;
    }

    SmallVector<MachineInstr *, 128> NewOrder;
    NewOrder.reserve(Units.size() * 5);
    NewOrder.push_back(Units.front().Load0);
    NewOrder.push_back(Units.front().Load1);
    for (size_t I = 1; I < Units.size(); ++I) {
      NewOrder.push_back(Units[I].Load0);
      NewOrder.push_back(Units[I].Load1);
      NewOrder.push_back(Units[I - 1].Arith0);
      NewOrder.push_back(Units[I - 1].Arith1);
      NewOrder.push_back(Units[I - 1].Store);
    }
    NewOrder.push_back(Units.back().Arith0);
    NewOrder.push_back(Units.back().Arith1);
    NewOrder.push_back(Units.back().Store);

    auto InsertPos = Units.front().Load0->getIterator();
    bool Changed = false;
    for (MachineInstr *MI : NewOrder) {
      if (MI->getIterator() != InsertPos) {
        MBB.splice(InsertPos, &MBB, MI->getIterator());
        Changed = true;
      }
      ++InsertPos;
    }
    return Changed;
  }

  bool stageTraceInBlock(MachineBasicBlock &MBB, const RISCVInstrInfo &TII,
                         const MachineRegisterInfo &MRI) const {
    bool Changed = false;
    for (auto It = MBB.begin(); It != MBB.end();) {
      TraceUnit FirstUnit;
      MachineBasicBlock::iterator NextIt;
      if (!matchUnit(It, MBB.end(), TII, FirstUnit, NextIt)) {
        ++It;
        continue;
      }

      SmallVector<TraceUnit, 32> Units;
      Units.push_back(FirstUnit);
      auto ScanIt = NextIt;
      while (ScanIt != MBB.end() && Units.size() < StarbugVLIWTraceMaxUnits) {
        TraceUnit U;
        MachineBasicBlock::iterator UNext;
        if (!matchUnit(ScanIt, MBB.end(), TII, U, UNext))
          break;
        Units.push_back(U);
        ScanIt = UNext;
      }

      if (Units.size() >= StarbugVLIWTraceMinUnits &&
          countDefVRegsWithinLimit(Units, MRI) &&
          verifyAndStage(Units, MBB, MRI)) {
        Changed = true;
        It = ScanIt;
        continue;
      }

      ++It;
    }
    return Changed;
  }
};

} // namespace

char StarbugVLIWTraceScheduler::ID = 0;

INITIALIZE_PASS(StarbugVLIWTraceScheduler, "starbug-vliw-trace-scheduler",
                STARBUG_VLIW_TRACE_SCHEDULER_NAME, false, false)

FunctionPass *llvm::createStarbugVLIWTraceSchedulerPass() {
  return new StarbugVLIWTraceScheduler();
}
