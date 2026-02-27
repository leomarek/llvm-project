#include "StarbugVLIWConfig.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <string>

using namespace llvm;
using namespace llvm::RISCVVLIW;

static cl::opt<unsigned> StarbugVLanes(
    "starbug-vliw-lanes", cl::init(4), cl::Hidden,
    cl::desc("Override Starbug VLIW lane count"));

static cl::opt<bool> StarbugVForceUnroll(
    "starbug-vliw-force-unroll", cl::init(true), cl::Hidden,
    cl::desc("Force aggressive loop unrolling for Starbug VLIW"));

static cl::opt<unsigned> StarbugVUnrollFactor(
    "starbug-vliw-unroll-factor", cl::init(32), cl::Hidden,
    cl::desc("Preferred loop unroll factor for Starbug VLIW"));

static cl::opt<unsigned> StarbugVMaxUnroll(
    "starbug-vliw-max-unroll", cl::init(128), cl::Hidden,
    cl::desc("Maximum unroll factor allowed for Starbug VLIW"));

static cl::opt<std::string> StarbugVLaneOpClasses(
    "starbug-vliw-lane-op-classes", cl::init(""), cl::Hidden,
    cl::desc("Override per-lane op classes. Format: "
             "lane:class[,class];lane:class[,class]. Example: "
             "0:any;1:alu_addsub,alu_muldiv,alu_shift"));

static cl::opt<bool> StarbugVEmitSingleHints(
    "starbug-vliw-emit-single-hints", cl::init(false), cl::Hidden,
    cl::desc("Emit STARBUG_BUNDLE_HINT for single-instruction packets"));

LaneConfig::LaneConfig() = default;

LaneConfig::LaneConfig(unsigned LaneId, unsigned NumClasses)
    : LaneId(LaneId), AllowedClasses(NumClasses, false) {}

static LaneConfig makeLane0() {
  LaneConfig L(0, static_cast<unsigned>(OpClass::Last));
  L.AllowedClasses.set(static_cast<unsigned>(OpClass::Any));
  return L;
}

static LaneConfig makeALULane(unsigned LaneId) {
  LaneConfig L(LaneId, static_cast<unsigned>(OpClass::Last));
  L.AllowedClasses.set(static_cast<unsigned>(OpClass::ALUAddSub));
  L.AllowedClasses.set(static_cast<unsigned>(OpClass::ALUMulDiv));
  L.AllowedClasses.set(static_cast<unsigned>(OpClass::ALUShift));
  return L;
}

static bool applyClassToken(StringRef Token, BitVector &Allowed) {
  Token = Token.trim();
  if (Token.empty())
    return true;
  const std::string Lower = Token.lower();
  const StringRef Key = Lower;

  auto setClass = [&](OpClass C) {
    Allowed.set(static_cast<unsigned>(C));
  };

  if (Key == "any" || Key == "all")
    setClass(OpClass::Any);
  else if (Key == "alu" || Key == "int") {
    setClass(OpClass::ALUAddSub);
    setClass(OpClass::ALUMulDiv);
    setClass(OpClass::ALUShift);
  } else if (Key == "alu_addsub" || Key == "addsub" || Key == "add" ||
             Key == "sub")
    setClass(OpClass::ALUAddSub);
  else if (Key == "alu_muldiv" || Key == "muldiv" || Key == "mul" ||
           Key == "div")
    setClass(OpClass::ALUMulDiv);
  else if (Key == "alu_shift" || Key == "shift" || Key == "sh")
    setClass(OpClass::ALUShift);
  else if (Key == "load" || Key == "ld")
    setClass(OpClass::Load);
  else if (Key == "store" || Key == "st")
    setClass(OpClass::Store);
  else if (Key == "branch" || Key == "br")
    setClass(OpClass::Branch);
  else if (Key == "compare" || Key == "cmp")
    setClass(OpClass::Compare);
  else if (Key == "move" || Key == "mv" || Key == "li" ||
           Key == "copy")
    setClass(OpClass::Move);
  else if (Key == "csr")
    setClass(OpClass::CSR);
  else
    return false;

  return true;
}

static bool parseLaneClassList(StringRef LaneClasses, LaneConfig &Lane) {
  Lane.AllowedClasses.reset();

  SmallVector<StringRef, 16> ClassTokens;
  LaneClasses.split(ClassTokens, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef Token : ClassTokens) {
    if (!applyClassToken(Token, Lane.AllowedClasses))
      return false;
  }

  return true;
}

static void ensureLaneExists(StarbugVLIWConfig &Cfg, unsigned LaneId) {
  for (unsigned I = Cfg.Lanes.size(); I <= LaneId; ++I) {
    if (I == 0)
      Cfg.Lanes.push_back(makeLane0());
    else
      Cfg.Lanes.push_back(makeALULane(I));
  }
}

static bool applyLaneOpClassOverrides(StarbugVLIWConfig &Cfg, StringRef Spec) {
  Spec = Spec.trim();
  if (Spec.empty())
    return true;

  SmallVector<StringRef, 8> LaneSpecs;
  Spec.split(LaneSpecs, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef LaneSpec : LaneSpecs) {
    LaneSpec = LaneSpec.trim();
    if (LaneSpec.empty())
      continue;

    auto Pair = LaneSpec.split(':');
    StringRef LaneIdStr = Pair.first.trim();
    StringRef ClassesStr = Pair.second.trim();
    if (LaneIdStr.empty() || ClassesStr.empty())
      return false;

    unsigned LaneId = 0;
    if (LaneIdStr.getAsInteger(10, LaneId))
      return false;

    ensureLaneExists(Cfg, LaneId);
    Cfg.Lanes[LaneId].LaneId = LaneId;
    if (!parseLaneClassList(ClassesStr, Cfg.Lanes[LaneId]))
      return false;
  }

  Cfg.NumLanes = std::max(Cfg.NumLanes, static_cast<unsigned>(Cfg.Lanes.size()));
  return true;
}

StarbugVLIWConfig StarbugVLIWConfig::getDefault() {
  StarbugVLIWConfig Cfg;
  Cfg.NumLanes = 4;

  Cfg.Lanes.clear();
  Cfg.Lanes.push_back(makeLane0());
  Cfg.Lanes.push_back(makeALULane(1));
  Cfg.Lanes.push_back(makeALULane(2));
  Cfg.Lanes.push_back(makeALULane(3));

  Cfg.Unroll.ForceUnroll = true;
  Cfg.Unroll.DefaultUnrollFactor = 32;
  Cfg.Unroll.MaxUnrollFactor = 128;
  Cfg.Unroll.PreferUnrollAndJam = true;
  Cfg.Unroll.SpillPressureGuard = true;

  Cfg.Scheduler.PrioritizePointerBumps = true;
  Cfg.Scheduler.PrioritizeReadyLoads = true;
  Cfg.Scheduler.AllowShortPackets = true;
  Cfg.Scheduler.EmitSingleInstructionHints = false;

  return Cfg;
}

StarbugVLIWConfig StarbugVLIWConfig::fromCommandLine() {
  StarbugVLIWConfig Cfg = getDefault();

  Cfg.NumLanes = std::max(1u, StarbugVLanes.getValue());
  Cfg.Unroll.ForceUnroll = StarbugVForceUnroll;
  Cfg.Unroll.DefaultUnrollFactor = StarbugVUnrollFactor;
  Cfg.Unroll.MaxUnrollFactor = StarbugVMaxUnroll;
  if (Cfg.Unroll.DefaultUnrollFactor == 0)
    Cfg.Unroll.DefaultUnrollFactor = 1;
  Cfg.Unroll.MaxUnrollFactor =
      std::max(Cfg.Unroll.MaxUnrollFactor, Cfg.Unroll.DefaultUnrollFactor);
  Cfg.Scheduler.EmitSingleInstructionHints = StarbugVEmitSingleHints;

  // TODO: Replace this with generated config loading from YAML-derived data.
  // For now, expand lane vector using lane0-any + ALU-only lanes.
  Cfg.Lanes.clear();
  Cfg.Lanes.push_back(makeLane0());
  for (unsigned Lane = 1; Lane < Cfg.NumLanes; ++Lane)
    Cfg.Lanes.push_back(makeALULane(Lane));

  if (!applyLaneOpClassOverrides(Cfg, StarbugVLaneOpClasses)) {
    errs() << "warning: ignoring invalid -starbug-vliw-lane-op-classes spec: "
           << StarbugVLaneOpClasses << '\n';
  }

  return Cfg;
}

bool StarbugVLIWConfig::isLaneLegal(unsigned Lane, OpClass Class) const {
  if (Lane >= Lanes.size())
    return false;

  const auto &Allowed = Lanes[Lane].AllowedClasses;
  const unsigned AnyClass = static_cast<unsigned>(OpClass::Any);
  const unsigned RequestedClass = static_cast<unsigned>(Class);

  if (AnyClass < Allowed.size() && Allowed.test(AnyClass))
    return true;

  return RequestedClass < Allowed.size() && Allowed.test(RequestedClass);
}
