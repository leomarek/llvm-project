#ifndef LLVM_LIB_TARGET_RISCV_STARBUGVLIWCONFIG_H
#define LLVM_LIB_TARGET_RISCV_STARBUGVLIWCONFIG_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace llvm {

namespace RISCVVLIW {

enum class OpClass : uint8_t {
  Any = 0,
  ALUAddSub,
  ALUMulDiv,
  ALUShift,
  Load,
  Store,
  Branch,
  Compare,
  Move,
  CSR,
  Last
};

struct LaneConfig {
  unsigned LaneId = 0;
  BitVector AllowedClasses;

  LaneConfig();
  LaneConfig(unsigned LaneId, unsigned NumClasses);
};

struct UnrollPolicy {
  bool ForceUnroll = true;
  unsigned DefaultUnrollFactor = 32;
  unsigned MaxUnrollFactor = 128;
  bool PreferUnrollAndJam = true;
  bool SpillPressureGuard = true;
};

struct SchedulerPolicy {
  bool PrioritizePointerBumps = true;
  bool PrioritizeReadyLoads = true;
  unsigned LatencyWeightLoad = 3;
  unsigned LatencyWeightMulDiv = 2;
  unsigned LatencyWeightStore = 1;
  bool AllowShortPackets = true;
  bool EmitSingleInstructionHints = false;
};

struct StarbugVLIWConfig {
  unsigned NumLanes = 4;
  SmallVector<LaneConfig, 8> Lanes;
  UnrollPolicy Unroll;
  SchedulerPolicy Scheduler;

  static StarbugVLIWConfig getDefault();
  static StarbugVLIWConfig fromCommandLine();

  bool isLaneLegal(unsigned Lane, OpClass Class) const;
  unsigned maxBundleWidth() const { return NumLanes; }
};

} // namespace RISCVVLIW

} // namespace llvm

#endif
