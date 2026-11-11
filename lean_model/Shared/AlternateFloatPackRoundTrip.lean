import AlternateFloatSpec
set_option maxHeartbeats 0
set_option maxRecDepth 100000
set_option exponentiation.threshold 4096
namespace GB300Math.AlternateFloat

theorem e4m3_normal_pack_roundtrip : ∀ s : Bool, ∀ e : Fin 16, ∀ q : Fin 17,
    1 ≤ e.val → 8 ≤ q.val → (e.val-1)*8+q.val ≤ 126 →
    decode "e4m3" ((if s then 128 else 0)+(e.val-1)*8+q.val) =
      some (.finite s (q.val*2^(e.val+502))) := by decide +kernel

theorem e5m2_normal_pack_roundtrip : ∀ s : Bool, ∀ e : Fin 31, ∀ q : Fin 9,
    1 ≤ e.val → 4 ≤ q.val → (e.val-1)*4+q.val ≤ 123 →
    decode "e5m2" ((if s then 128 else 0)+(e.val-1)*4+q.val) =
      some (.finite s (q.val*2^(e.val+495))) := by decide +kernel

theorem e4m3_subnormal_pack_roundtrip : ∀ s : Bool, ∀ q : Fin 9,
    decode "e4m3" ((if s then 128 else 0)+q.val) =
      some (.finite s (q.val*2^503)) := by decide +kernel

theorem e5m2_subnormal_pack_roundtrip : ∀ s : Bool, ∀ q : Fin 5,
    decode "e5m2" ((if s then 128 else 0)+q.val) =
      some (.finite s (q.val*2^496)) := by decide +kernel

#print axioms e4m3_normal_pack_roundtrip
#print axioms e5m2_normal_pack_roundtrip
#print axioms e4m3_subnormal_pack_roundtrip
#print axioms e5m2_subnormal_pack_roundtrip
end GB300Math.AlternateFloat
