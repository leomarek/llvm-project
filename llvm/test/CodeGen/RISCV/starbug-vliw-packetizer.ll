; Packetizer safety tests for the STARBUG VLIW extension.
;
; A STARBUG bundle hint (c.li zero, N) asserts that the next N instructions are
; mutually independent and lane-legal. Hardware does not check this, so a wrong
; hint is a silent wrong answer. These tests pin the two directions that matter:
; a hint must appear where parallelism genuinely exists, and must not appear
; where it does not.
;
; RUN: llc -mtriple=riscv32 -mcpu=starbug-vliw -mattr=+m,+c < %s | FileCheck %s

; A strictly serial dependence chain has no independent pairs anywhere, so the
; packetizer must not emit a hint. Before the scheduling fix this function was
; one of the shapes that produced a bundle containing its own operands.
define i32 @serial_chain(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: serial_chain:
; CHECK-NOT: c.li zero
; CHECK: ret
  %t1 = add i32 %a, %b
  %t2 = mul i32 %t1, %a
  %t3 = add i32 %t2, %t1
  %t4 = mul i32 %t3, %t2
  %t5 = add i32 %t4, %t3
  ret i32 %t5
}

; Four mutually independent multiplies over disjoint operands: the packetizer
; should find real parallelism here and emit at least one hint.
define i32 @independent_ops(i32 %a, i32 %b, i32 %c, i32 %d,
                            i32 %e, i32 %f, i32 %g, i32 %h) nounwind {
; CHECK-LABEL: independent_ops:
; CHECK: c.li zero
  %p1 = mul i32 %a, %b
  %p2 = mul i32 %c, %d
  %p3 = mul i32 %e, %f
  %p4 = mul i32 %g, %h
  %s1 = add i32 %p1, %p2
  %s2 = add i32 %p3, %p4
  %r  = add i32 %s1, %s2
  ret i32 %r
}

; A hint may never declare more than four instructions: ifu.sv forms a bundle
; only for 1 <= imm <= 4 and ignores anything larger outright.
define void @bundle_width_cap(ptr %p, i32 %a, i32 %b, i32 %c, i32 %d,
                              i32 %e, i32 %f) nounwind {
; CHECK-LABEL: bundle_width_cap:
; CHECK-NOT: c.li zero, 5
; CHECK-NOT: c.li zero, 6
; CHECK-NOT: c.li zero, 7
; CHECK-NOT: c.li zero, 8
  %x1 = add i32 %a, %b
  %x2 = add i32 %c, %d
  %x3 = add i32 %e, %f
  %x4 = xor i32 %a, %c
  %x5 = xor i32 %b, %d
  %x6 = xor i32 %e, %a
  %y1 = add i32 %x1, %x2
  %y2 = add i32 %x3, %x4
  %y3 = add i32 %x5, %x6
  %z  = add i32 %y1, %y2
  %w  = add i32 %z, %y3
  store i32 %w, ptr %p
  ret void
}
