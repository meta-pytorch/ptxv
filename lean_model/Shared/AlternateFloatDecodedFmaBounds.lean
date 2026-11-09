import AlternateFloatOperationBounds
set_option maxHeartbeats 0
set_option maxRecDepth 100000
set_option exponentiation.threshold 4096
namespace GB300Math.AlternateFloatFloatLib
open GB300Math.AlternateFloat

def sourceFormat (i : Fin 7) : String :=
  ["e4m3", "e5m2", "e3m2", "e2m3", "e2m1", "e2m1p4", "ue8m0"][i.val]!

def scalarFma (dst af bf : String) (a b c : Nat) : Option ℚ := do
  let x ← decode af a
  let y ← decode bf b
  let z ← decode dst c
  rational (decode dst (encode dst false (add (mul x y) z)))

theorem decoded_finite_grid (fmt : Fin 7) (code : Fin 256) (s : Bool) (m : Nat)
    (h : decode (sourceFormat fmt) code.val = some (.finite s m)) : m % 2^385 = 0 := by
  have hg := decoded_inputs_grid fmt code
  change finiteOnGrid (decode (sourceFormat fmt) code.val) at hg
  simpa [h, finiteOnGrid] using hg

theorem e4m3_decoded_fma_bound (af bf : Fin 7) (a b c : Fin 256)
    (sa sb sc : Bool) (ma mb mc : Nat)
    (ha : decode (sourceFormat af) a.val = some (.finite sa ma))
    (hb : decode (sourceFormat bf) b.val = some (.finite sb mb))
    (hc : decode "e4m3" c.val = some (.finite sc mc))
    (hmax : |((if sa then -1 else 1)*(ma : ℚ)/unit)*
      ((if sb then -1 else 1)*(mb : ℚ)/unit)+(if sc then -1 else 1)*(mc : ℚ)/unit| ≤ 448) :
    let ideal : ℚ := ((if sa then -1 else 1)*(ma : ℚ)/unit)*
      ((if sb then -1 else 1)*(mb : ℚ)/unit)+(if sc then -1 else 1)*(mc : ℚ)/unit
    ∃ y : ℚ, scalarFma "e4m3" (sourceFormat af) (sourceFormat bf) a.val b.val c.val = some y ∧
      |y-ideal| ≤ max (|ideal|/17) (1/1024) := by
  have r := e4m3_fma_ideal_bound sa sb sc ma mb mc
    (decoded_finite_grid af a sa ma ha) (decoded_finite_grid bf b sb mb hb) hmax
  simpa [scalarFma, ha, hb, hc] using r

theorem e5m2_decoded_fma_bound (af bf : Fin 7) (a b c : Fin 256)
    (sa sb sc : Bool) (ma mb mc : Nat)
    (ha : decode (sourceFormat af) a.val = some (.finite sa ma))
    (hb : decode (sourceFormat bf) b.val = some (.finite sb mb))
    (hc : decode "e5m2" c.val = some (.finite sc mc))
    (hmax : |((if sa then -1 else 1)*(ma : ℚ)/unit)*
      ((if sb then -1 else 1)*(mb : ℚ)/unit)+(if sc then -1 else 1)*(mc : ℚ)/unit| ≤ 57344) :
    let ideal : ℚ := ((if sa then -1 else 1)*(ma : ℚ)/unit)*
      ((if sb then -1 else 1)*(mb : ℚ)/unit)+(if sc then -1 else 1)*(mc : ℚ)/unit
    ∃ y : ℚ, scalarFma "e5m2" (sourceFormat af) (sourceFormat bf) a.val b.val c.val = some y ∧
      |y-ideal| ≤ max (|ideal|/9) (1/131072) := by
  have r := e5m2_fma_ideal_bound sa sb sc ma mb mc
    (decoded_finite_grid af a sa ma ha) (decoded_finite_grid bf b sb mb hb) hmax
  simpa [scalarFma, ha, hb, hc] using r

theorem e4m3_fma_relative_sharp : scalarFma "e4m3" "e4m3" "e4m3" 56 56 24 = some 1 := by decide +kernel
theorem e5m2_fma_relative_sharp : scalarFma "e5m2" "e5m2" "e5m2" 60 60 48 = some 1 := by decide +kernel
theorem e4m3_fma_subnormal_sharp : scalarFma "e4m3" "e4m3" "e4m3" 1 48 0 = some 0 := by decide +kernel
theorem e5m2_fma_subnormal_sharp : scalarFma "e5m2" "e5m2" "e5m2" 1 56 0 = some 0 := by decide +kernel

#print axioms decoded_finite_grid
#print axioms e4m3_decoded_fma_bound
#print axioms e5m2_decoded_fma_bound
#print axioms e4m3_fma_relative_sharp
#print axioms e5m2_fma_relative_sharp
#print axioms e4m3_fma_subnormal_sharp
#print axioms e5m2_fma_subnormal_sharp
end GB300Math.AlternateFloatFloatLib
