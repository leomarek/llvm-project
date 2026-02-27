// Starbug-specific pseudo instruction to annotate packet size before emission.
let hasSideEffects = 1, mayLoad = 0, mayStore = 0, isPseudo = 1 in {
  def STARBUG_BUNDLE_HINT : Pseudo<(outs), (ins i32imm:$len), []>;
}
