import GB300Math.MultisampleFetch
import GB300Math.TensorEncodingBounds

namespace GB300Math.MultisamplePayloadExact
open Tensor TensorEncodingBounds

theorem decode_payload (n : Nat) (hn : 0 < n) (bound : n < 2^23) :
    decode (MultisampleFetch.encode n) 23 8 =
      ⟨(n*2^(23-n.log2) : Nat), (n.log2 : Int)-39⟩ := by
  have hn0 : n ≠ 0 := by omega
  have top : n.log2 < 23 := (Nat.log2_lt hn0).mpr bound
  have hm := normalized_mantissa_bounds n hn0
  have he : 0 < n.log2+111 ∧ n.log2+111 < 255 := by omega
  have h := decode_normal false (normalizedMantissa n) (n.log2+111) hm he
  have htop : ¬ n.log2 ≥ 23 := by omega
  simp only [normalizedMantissa, htop, packNormal, Bool.false_eq_true,
    ↓reduceIte, Nat.zero_add] at h
  change decode (MultisampleFetch.encode n) 23 8 =
    ⟨(n*2^(23-n.log2) : Nat), ((n.log2+111 : Nat) : Int)-150⟩ at h
  rewrite [h]
  congr 1
  omega

theorem exact_common_units (n : Nat) (hn : 0 < n) (bound : n < 2^23) :
    (decode (MultisampleFetch.encode n) 23 8).significand * (2^n.log2 : Nat) =
      (n : Int) * (2^23 : Nat) := by
  rewrite [decode_payload n hn bound]
  dsimp only
  have top : n.log2 < 23 := (Nat.log2_lt (by omega : n ≠ 0)).mpr bound
  have powers : 23-n.log2+n.log2=23 := by omega
  norm_cast
  simp only [Nat.mul_assoc, ← Nat.pow_add, powers]

theorem fixture_is_exact (x y layer sample : Nat)
    (hx : x < 16) (hy : y < 16) (hl : layer < 2) (hs : sample < 8) :
    let n := MultisampleFetch.payload x y layer sample;
    (decode (MultisampleFetch.encode n) 23 8).significand * (2^n.log2 : Nat) =
      (n : Int) * (2^23 : Nat) := by
  apply exact_common_units <;> unfold MultisampleFetch.payload <;> omega

end GB300Math.MultisamplePayloadExact
