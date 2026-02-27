#ifndef LLVM_LIB_TARGET_RISCV_STARBUGVLIWPACKETIZER_H
#define LLVM_LIB_TARGET_RISCV_STARBUGVLIWPACKETIZER_H

#include "StarbugVLIWConfig.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {

class RISCVSubtarget;

class StarbugVLIWPacketizer : public MachineFunctionPass {
public:
  static char ID;

  StarbugVLIWPacketizer();
  StringRef getPassName() const override;
  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  RISCVVLIW::StarbugVLIWConfig Config;

  bool packetizeBasicBlock(MachineBasicBlock &MBB, const RISCVSubtarget &ST);
};

FunctionPass *createStarbugVLIWPacketizerPass();

} // namespace llvm

#endif
