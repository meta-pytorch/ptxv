/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Theorem

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false

namespace MemoryWrite

def atomicByteWord (bytes : List U8) : U64 :=
  ⟨(BitVec.fromLEBytes (bytes.map U8.bv)).setWidth 64⟩

@[step]
theorem rust_atomic_encode_spec (value : U64) :
    atomic_bytes.encode value
      ⦃ bytes => bytes.val = value.bv.toLEBytes.map (@UScalar.mk .U8) ⦄ := by
  unfold atomic_bytes.encode
  simp [core.num.U64.to_le_bytes, Array.from_val]

theorem rust_atomic_decode_two (bytes : Slice U8) (a b : U8)
    (hbytes : bytes.val = [a, b]) :
    atomic_bytes.decode bytes ⦃ result => result = .Ok (atomicByteWord bytes.val) ⦄ := by
  unfold atomic_bytes.decode
  have hlen : (Slice.len bytes).val = 2 := by simp [Slice.len, hbytes]
  simp only [hlen]
  step with Slice.index_usize_spec bytes 0#usize (by simp [hbytes]) as ⟨first, hfirst⟩
  step with Slice.index_usize_spec bytes 1#usize (by simp [hbytes]) as ⟨second, hsecond⟩
  simp only [hbytes, List.getElem_cons_zero, List.getElem_cons_succ] at hfirst hsecond
  subst first second
  simp only [WP.spec_ok, lift, bind_ok, core.result.Result.Ok.injEq]
  rw [hbytes]
  rfl

theorem rust_atomic_decode_four (bytes : Slice U8) (a b c d : U8)
    (hbytes : bytes.val = [a, b, c, d]) :
    atomic_bytes.decode bytes ⦃ result => result = .Ok (atomicByteWord bytes.val) ⦄ := by
  unfold atomic_bytes.decode
  have hlen : (Slice.len bytes).val = 4 := by simp [Slice.len, hbytes]
  simp only [hlen]
  step with Slice.index_usize_spec bytes 0#usize (by simp [hbytes]) as ⟨first, hfirst⟩
  step with Slice.index_usize_spec bytes 1#usize (by simp [hbytes]) as ⟨second, hsecond⟩
  step with Slice.index_usize_spec bytes 2#usize (by simp [hbytes]) as ⟨third, hthird⟩
  step with Slice.index_usize_spec bytes 3#usize (by simp [hbytes]) as ⟨fourth, hfourth⟩
  simp only [hbytes, List.getElem_cons_zero, List.getElem_cons_succ]
    at hfirst hsecond hthird hfourth
  subst first second third fourth
  simp only [WP.spec_ok, lift, bind_ok, core.result.Result.Ok.injEq]
  rw [hbytes]
  rfl

theorem rust_atomic_decode_eight (bytes : Slice U8) (a b c d e f g h : U8)
    (hbytes : bytes.val = [a, b, c, d, e, f, g, h]) :
    atomic_bytes.decode bytes ⦃ result => result = .Ok (atomicByteWord bytes.val) ⦄ := by
  unfold atomic_bytes.decode
  have hlen : (Slice.len bytes).val = 8 := by simp [Slice.len, hbytes]
  simp only [hlen]
  step with Slice.index_usize_spec bytes 0#usize (by simp [hbytes]) as ⟨first, hfirst⟩
  step with Slice.index_usize_spec bytes 1#usize (by simp [hbytes]) as ⟨second, hsecond⟩
  step with Slice.index_usize_spec bytes 2#usize (by simp [hbytes]) as ⟨third, hthird⟩
  step with Slice.index_usize_spec bytes 3#usize (by simp [hbytes]) as ⟨fourth, hfourth⟩
  step with Slice.index_usize_spec bytes 4#usize (by simp [hbytes]) as ⟨fifth, hfifth⟩
  step with Slice.index_usize_spec bytes 5#usize (by simp [hbytes]) as ⟨sixth, hsixth⟩
  step with Slice.index_usize_spec bytes 6#usize (by simp [hbytes]) as ⟨seventh, hseventh⟩
  step with Slice.index_usize_spec bytes 7#usize (by simp [hbytes]) as ⟨eighth, heighth⟩
  simp only [hbytes, List.getElem_cons_zero, List.getElem_cons_succ]
    at hfirst hsecond hthird hfourth hfifth hsixth hseventh heighth
  subst first second third fourth fifth sixth seventh eighth
  simp only [WP.spec_ok, lift, bind_ok, core.result.Result.Ok.injEq]
  rw [hbytes]
  apply UScalar.eq_of_val_eq
  change (BitVec.fromLEBytes ([a, b, c, d, e, f, g, h].map U8.bv)).toNat =
    ((BitVec.fromLEBytes ([a, b, c, d, e, f, g, h].map U8.bv)).setWidth 64).toNat
  rw [BitVec.toNat_setWidth]
  exact (Nat.mod_eq_of_lt (BitVec.fromLEBytes ([a,b,c,d,e,f,g,h].map U8.bv)).isLt).symm

private theorem atomic_eight_shape (bytes : List U8) (hlen : bytes.length = 8) :
    ∃ a b c d e f g h, bytes = [a, b, c, d, e, f, g, h] := by
  obtain ⟨a, b, c, d, hfront⟩ := List.length_eq_four.mp
    (show (bytes.take 4).length = 4 by simp [hlen])
  obtain ⟨e, f, g, h, hback⟩ := List.length_eq_four.mp
    (show (bytes.drop 4).length = 4 by simp [hlen])
  have hbytes := List.take_append_drop 4 bytes
  rw [hfront, hback] at hbytes
  exact ⟨a, b, c, d, e, f, g, h, hbytes.symm⟩

def atomicDecodeModel (bytes : List U8) :
    core.result.Result U64 atomic_integer.AtomicError :=
  if bytes.length = 2 ∨ bytes.length = 4 ∨ bytes.length = 8 then
    .Ok (atomicByteWord bytes)
  else .Err .UnsupportedWidth

@[step]
theorem rust_atomic_decode_spec (bytes : Slice U8) :
    atomic_bytes.decode bytes ⦃ result => result = atomicDecodeModel bytes.val ⦄ := by
  by_cases htwo : bytes.val.length = 2
  · obtain ⟨a, b, hbytes⟩ := List.length_eq_two.mp htwo
    simpa only [atomicDecodeModel, htwo, true_or, ite_true] using
      rust_atomic_decode_two bytes a b hbytes
  by_cases hfour : bytes.val.length = 4
  · obtain ⟨a, b, c, d, hbytes⟩ := List.length_eq_four.mp hfour
    simpa only [atomicDecodeModel, hfour, true_or, or_true, ite_true] using
      rust_atomic_decode_four bytes a b c d hbytes
  by_cases height : bytes.val.length = 8
  · obtain ⟨a, b, c, d, e, f, g, h, hbytes⟩ := atomic_eight_shape bytes.val height
    simpa only [atomicDecodeModel, height, or_true, ite_true] using
      rust_atomic_decode_eight bytes a b c d e f g h hbytes
  · unfold atomic_bytes.decode
    have hlen : (Slice.len bytes).val = bytes.val.length := by simp [Slice.len]
    simp only [hlen]
    simp_all [atomicDecodeModel, WP.spec_ok]

theorem rust_atomic_decode_total (bytes : Slice U8) :
    atomic_bytes.decode bytes = .ok (atomicDecodeModel bytes.val) := by
  obtain ⟨result, hactual, hresult⟩ := spec_returns _ _ (rust_atomic_decode_spec bytes)
  simpa only [hresult] using hactual

theorem atomicByteWord_encode (value : U64) :
    atomicByteWord (value.bv.toLEBytes.map (@UScalar.mk .U8)) = value := by
  apply UScalar.eq_of_val_eq
  change ((BitVec.fromLEBytes
    ((value.bv.toLEBytes.map (@UScalar.mk .U8)).map U8.bv)).setWidth 64).toNat = value.bv.toNat
  rw [BitVec.toNat_setWidth]
  have hmap : (value.bv.toLEBytes.map (@UScalar.mk .U8)).map U8.bv = value.bv.toLEBytes := by
    simp only [List.map_map, Function.comp_def]
    exact List.map_id _
  rw [hmap, BitVec.fromLEBytes_toLEBytes (by decide), BitVec.toNat_cast]
  exact Nat.mod_eq_of_lt value.bv.isLt

theorem rust_atomic_codec_roundtrip (value : U64) :
    ∃ encoded, atomic_bytes.encode value = .ok encoded ∧
      ∀ bytes : Slice U8, bytes.val = encoded.val →
        atomic_bytes.decode bytes = .ok (.Ok value) := by
  obtain ⟨encoded, hactual, hencoded⟩ := spec_returns _ _ (rust_atomic_encode_spec value)
  refine ⟨encoded, hactual, ?_⟩
  intro bytes hbytes
  have hlen : bytes.val.length = 8 := by
    rw [hbytes]
    exact Array.property encoded
  rw [rust_atomic_decode_total]
  simp only [atomicDecodeModel, hlen, or_true, ite_true]
  rw [hbytes, hencoded, atomicByteWord_encode]

end MemoryWrite
