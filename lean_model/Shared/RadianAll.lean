import RadianTinyIdeal
namespace GB300Math.RadianAll
open FloatLib.Floats.Formats.BinaryInterchange
open Root32FloatLib RadianSigned
set_option maxRecDepth 16000
set_option maxHeartbeats 6000000
theorem sin_mode (ftz : Bool) (w : Nat) : Approx.evaluate "sin" ftz w=Approx.evaluate "sin" false w := rfl
theorem cos_mode (ftz : Bool) (w : Nat) : Approx.evaluate "cos" ftz w=Approx.evaluate "cos" false w := rfl

theorem finite_bound (ftz negative : Bool) (e : Fin 255) (f : Fin (2^23)) :
    ∃ x ys yc : Rat,
      value (word negative e.val f.val)=some x ∧
      Approx.evaluate "sin" ftz (word negative e.val f.val)=some (Mufu.sincos (Approx.mul (word negative e.val f.val) 0x3e22f983 true) false) ∧
      Approx.evaluate "cos" ftz (word negative e.val f.val)=some (Mufu.sincos (Approx.mul (word negative e.val f.val) 0x3e22f983 true) true) ∧
      value (Mufu.sincos (Approx.mul (word negative e.val f.val) 0x3e22f983 true) false)=some ys ∧
      value (Mufu.sincos (Approx.mul (word negative e.val f.val) 0x3e22f983 true) true)=some yc ∧
      |(ys:ℝ)-Real.sin (x:ℝ)|≤1/2^20+|(x:ℝ)|/2^22 ∧
      |(yc:ℝ)-Real.cos (x:ℝ)|≤1/2^20+|(x:ℝ)|/2^22 := by
  rw [sin_mode,cos_mode]
  by_cases he:e.val<4
  · obtain ⟨x,hx,hs,hc,hse,hce⟩:=RadianTinyIdeal.decoded_small_bound negative e.val f.val he f.isLt
    have hzero : (0:ℝ)≤|(x:ℝ)|/2^22 := by positivity
    refine ⟨x,0,1,hx,?_,?_,hs,hc,?_,?_⟩
    · rfl
    · rfl
    · norm_num only [Rat.cast_zero]
      nlinarith only[hse,hzero]
    · norm_num only [Rat.cast_one]
      nlinarith only[hce,hzero]
  · exact RadianSigned.operational_bound negative e.val f.val (by have h:=e.isLt;omega) f.isLt

theorem finite_word_count : 2*255*2^23=4278190080 := by decide +kernel
theorem added_tiny_word_count : 2*4*2^23=67108864 := by decide +kernel
#print axioms sin_mode
#print axioms cos_mode
#print axioms added_tiny_word_count
#print axioms finite_bound
#print axioms finite_word_count
end GB300Math.RadianAll
