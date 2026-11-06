import Log2Uniform
import Log2NearEncoding
import Log2TightTable
namespace GB300Math.Log2PtxRange
set_option maxHeartbeats 4000000
set_option maxRecDepth 10000

theorem near_range (e : Fin 254) (f : Fin (2^23)) (he : 125≤e.val ∧ e.val≤126) :
    -(2^36:Int)<Log2Uniform.fixedValue e f ∧ Log2Uniform.fixedValue e f<2^36 := by
  have h:=Log2Interpolation.interpolation_facts f.val f.isLt
  dsimp only at h
  dsimp [Log2Uniform.fixedValue]
  omega

theorem near_absolute_error (e : Fin 254) (f : Fin (2^23)) (he : 125≤e.val ∧ e.val≤126) :
    ∃ x y : Rat,
      Root32FloatLib.value ((1+e.val)*2^23+f.val)=some x ∧
      Root32FloatLib.value (Mufu.lg2 ((1+e.val)*2^23+f.val))=some y ∧
      0<x ∧ |(y:ℝ)-Real.log (x:ℝ)/Real.log 2|≤1/2^22 := by
  by_cases hone : (1+e.val)*2^23+f.val=0x3f800000
  · refine ⟨1,0,?_,?_,by norm_num,?_⟩
    · rw [hone];decide +kernel
    · rw [hone];decide +kernel
    · norm_num
  · obtain ⟨y,hy,hyerr⟩ := Log2NearEncoding.decoded_error (Log2Uniform.fixedValue e f) (near_range e f he)
    refine ⟨Log2Input.input e f,y,Log2Input.decoded e f,?_,Log2Input.positive e f,?_⟩
    · rwa [Log2Uniform.model_normal e f hone]
    have herr := Log2TightTable.table_error f.val f.isLt
    rw [Log2Input.ideal]
    have hid : (Log2Uniform.fixedValue e f:ℝ)/2^36=
        (e.val:ℝ)-126+(Mufu.interpolate MufuTables.lg2 f.val:ℝ)/2^36 := by
      dsimp [Log2Uniform.fixedValue]
      push_cast
      ring
    rw [hid] at hyerr
    have ht := abs_sub_le (y:ℝ)
      ((e.val:ℝ)-126+(Mufu.interpolate MufuTables.lg2 f.val:ℝ)/2^36)
      ((e.val:ℝ)-126+Real.log (1+(f.val:ℝ)/2^23)/Real.log 2)
    have herr' : |((e.val:ℝ)-126+(Mufu.interpolate MufuTables.lg2 f.val:ℝ)/2^36)-
        ((e.val:ℝ)-126+Real.log (1+(f.val:ℝ)/2^23)/Real.log 2)|≤177/1000000000 := by
      convert herr using 1; ring
    linarith

theorem raw_near_absolute_error (w : Nat) (hw : 126*2^23≤w ∧ w<128*2^23) :
    ∃ x y : Rat, Root32FloatLib.value w=some x ∧ Root32FloatLib.value (Mufu.lg2 w)=some y ∧
      0<x ∧ |(y:ℝ)-Real.log (x:ℝ)/Real.log 2|≤1/2^22 := by
  let e : Fin 254 := ⟨w/2^23-1,by omega⟩
  let f : Fin (2^23) := ⟨w%2^23,by omega⟩
  have hword : (1+e.val)*2^23+f.val=w := by dsimp [e,f];omega
  have h:=near_absolute_error e f (by dsimp [e];omega)
  rwa [hword] at h

#print axioms near_range
#print axioms near_absolute_error
#print axioms raw_near_absolute_error
end GB300Math.Log2PtxRange
