/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AtomicCompareExchange
import MemoryWrite.AtomicBytes

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite

def unknownAtomicByte (resultDefined : Bool) (value : U8) : PayloadByte :=
  ⟨value, none, true, resultDefined, none⟩

def atomicReplacementPayload (encoded : Array U8 8#usize) (resultDefined : Bool)
    (width : U8) : Slice PayloadByte :=
  Slice.from ((encoded.val.take width.val).map (unknownAtomicByte resultDefined)) (by
    have hlength := Array.property encoded
    have hmax : 8 ≤ Usize.max := by
      have hbounds := Usize.bounds_eq
      simp only [U32.max_eq, U64.max_eq] at hbounds
      omega
    simp only [List.length_map, List.length_take]
    have hlen : encoded.val.length = 8 := by simp
    omega)

@[simp]
theorem atomicReplacementPayload_length (encoded : Array U8 8#usize) (resultDefined : Bool) (width : U8)
    (hwidth : width.val ≤ 8) :
    (atomicReplacementPayload encoded resultDefined width).val.length = width.val := by
  have hlen : encoded.val.length = 8 := by simp
  simp [atomicReplacementPayload, hlen, Nat.min_eq_left hwidth]

theorem atomicReplacementPayload_at (encoded : Array U8 8#usize) (resultDefined : Bool) (width : U8)
    (index : Nat) (hwidth : width.val ≤ 8) (hindex : index < width.val) :
    payloadAt (atomicReplacementPayload encoded resultDefined width) index =
      unknownAtomicByte resultDefined (encoded.val[index]?.getD 0#u8) := by
  simp only [payloadAt, atomicReplacementPayload, Slice.from_val,
    List.getElem?_map, List.getElem?_take]
  simp only [hindex, ite_true]
  have hbound : index < encoded.val.length := by simp; omega
  rw [List.getElem?_eq_getElem hbound]
  rfl

theorem atomicReplacementPayload_words (encoded : Array U8 8#usize) (resultDefined : Bool) (width : U8) (index : Nat) :
    (payloadAt (atomicReplacementPayload encoded resultDefined width) index).word = none := by
  simp only [payloadAt, atomicReplacementPayload, Slice.from_val,
    List.getElem?_map, List.getElem?_take]
  split <;> cases encoded.val[index]? <;> rfl

theorem wordAfter_unknown_atomic (cell : Cell) (destination : Key)
    (encoded : Array U8 8#usize) (resultDefined : Bool) (width : U8) (hwidth : width.val ≤ 8) :
    wordAfter cell destination (atomicReplacementPayload encoded resultDefined width) =
      if Overlaps destination width.val cell.key then none else cell.word := by
  have hlength := atomicReplacementPayload_length encoded resultDefined width hwidth
  by_cases hcovered : CoversWord destination width.val cell.key
  · have hoverlap : Overlaps destination width.val cell.key := by
      obtain ⟨hregion, hstart, hend⟩ := hcovered
      refine ⟨hregion, ?_, ?_, ?_⟩ <;> omega
    simp [wordAfter, hlength, hcovered, hoverlap, atomicReplacementPayload_words]
  · simp [wordAfter, hlength, hcovered]

structure AtomicWrittenCell (before after : Cell) (destination : Key)
    (encoded : Array U8 8#usize) (resultDefined : Bool) (width : U8) : Prop where
  key : after.key = before.key
  history : after.word = if Overlaps destination width.val before.key then none else before.word
  bytes :
    if Touches destination width.val before.key then
      after.value = encoded.val[before.key.offset.val - destination.offset.val]?.getD 0#u8 ∧
      after.origin = none ∧ after.tainted = true ∧ after.defined = resultDefined ∧ after.written = true
    else after.value = before.value ∧ after.origin = before.origin ∧
      after.tainted = before.tainted ∧ after.defined = before.defined ∧ after.written = before.written

def AtomicWriteSafety (before after : Slice Cell) (destination : Key)
    (encoded : Array U8 8#usize) (resultDefined : Bool) (width : U8) : Prop :=
  List.Forall₂ (fun old next => AtomicWrittenCell old next destination encoded resultDefined width)
    before.val after.val

theorem atomic_model_cell_safety (before : Cell) (destination : Key)
    (encoded : Array U8 8#usize) (resultDefined : Bool) (width : U8) (hwidth : width.val ≤ 8) :
    AtomicWrittenCell before (modelCell before destination (atomicReplacementPayload encoded resultDefined width))
      destination encoded resultDefined width := by
  have hlength := atomicReplacementPayload_length encoded resultDefined width hwidth
  have hspec := modelCell_spec before destination (atomicReplacementPayload encoded resultDefined width)
  refine ⟨hspec.key, ?_, ?_⟩
  · rw [hspec.word_effect]
    exact wordAfter_unknown_atomic before destination encoded resultDefined width hwidth
  · by_cases htouched : Touches destination width.val before.key
    · have hindex : before.key.offset.val - destination.offset.val < width.val := by
        obtain ⟨_, hstart, hend⟩ := htouched
        omega
      simpa only [hlength, htouched, ite_true,
        atomicReplacementPayload_at encoded resultDefined width _ hwidth hindex, unknownAtomicByte] using hspec.byte_effect
    · simpa only [hlength, htouched, ite_false] using hspec.byte_effect

theorem atomic_map_safety (before : List Cell) (destination : Key)
    (encoded : Array U8 8#usize) (resultDefined : Bool) (width : U8) (hwidth : width.val ≤ 8) :
    List.Forall₂ (fun old next => AtomicWrittenCell old next destination encoded resultDefined width)
      before (before.map (fun cell => modelCell cell destination (atomicReplacementPayload encoded resultDefined width))) := by
  induction before with
  | nil => exact .nil
  | cons cell cells ih => exact .cons (atomic_model_cell_safety cell destination encoded resultDefined width hwidth) ih

theorem rust_atomic_replacement_write_safety (before : Slice Cell) (destination : Key)
    (encoded : Array U8 8#usize) (resultDefined : Bool) (width : U8) (hwidth : width.val ≤ 8) :
    ∃ after,
      apply_write_effect before destination (atomicReplacementPayload encoded resultDefined width) = .ok after ∧
      AtomicWriteSafety before after destination encoded resultDefined width := by
  obtain ⟨after, hactual, hmap⟩ := spec_returns _ _
    (apply_write_effect_spec before destination (atomicReplacementPayload encoded resultDefined width))
  refine ⟨after, hactual, ?_⟩
  unfold AtomicWriteSafety
  rw [hmap]
  exact atomic_map_safety before.val destination encoded resultDefined width hwidth

/-- Composition on existing cells; the caller supplies result definedness. -/
def atomicCellCommit {F : Type} (implementation : atomic_exec.FloatingArithmetic F)
    (input : atomic_exec.AtomicInput) (resultDefined : Bool) (before : Slice Cell) (destination : Key) :
    Result (core.result.Result (atomic_exec.AtomicOutcome × Slice Cell) atomic_integer.AtomicError) := do
  let result ← atomic_exec.execute implementation input
  match result with
  | .Err error => .ok (.Err error)
  | .Ok outcome =>
    if outcome.did_write then
      let encoded ← atomic_bytes.encode outcome.replacement
      let after ← apply_write_effect before destination (atomicReplacementPayload encoded resultDefined outcome.bytes)
      .ok (.Ok (outcome, after))
    else
      let after ← words.install_word before ⟨destination, none⟩
      .ok (.Ok (outcome, after))

theorem rust_successful_cas_cell_commit {F : Type}
    (implementation : atomic_exec.FloatingArithmetic F) (input : atomic_exec.AtomicInput)
    (resultDefined : Bool) (before : Slice Cell) (destination : Key)
    (hoperation : input.op = .Integer .Cas) (hwidth : input.bytes = 4#u8 ∨ input.bytes = 8#u8)
    (hcompare : integerNormalize input.bytes input.old = integerNormalize input.bytes input.compare) :
    ∃ outcome encoded after,
      atomic_exec.execute implementation input = .ok (.Ok outcome) ∧
      AtomicCompareExchangeContract input outcome ∧ outcome.did_write = true ∧
      atomic_bytes.encode outcome.replacement = .ok encoded ∧
      atomicCellCommit implementation input resultDefined before destination = .ok (.Ok (outcome, after)) ∧
      apply_write_effect before destination (atomicReplacementPayload encoded resultDefined input.bytes) = .ok after ∧
      AtomicWriteSafety before after destination encoded resultDefined input.bytes := by
  obtain ⟨outcome, hexecute, hcontract⟩ := rust_atomic_compare_exchange_progress implementation
    input .Cas hoperation (Or.inl rfl) hwidth
  have hswapped : outcome.swapped = true := by simpa [hcompare] using hcontract.swapped
  have hwrites : outcome.did_write = true := by
    simpa [hoperation, atomicIsCas, hswapped] using hcontract.controls.writes
  have hbound : input.bytes.val ≤ 8 := by rcases hwidth with hwidth | hwidth <;> simp [hwidth]
  obtain ⟨encoded, hencode, _⟩ := spec_returns _ _ (rust_atomic_encode_spec outcome.replacement)
  obtain ⟨after, hwrite, hsafety⟩ :=
    rust_atomic_replacement_write_safety before destination encoded resultDefined input.bytes hbound
  refine ⟨outcome, encoded, after, hexecute, hcontract, hwrites, hencode, ?_, hwrite, hsafety⟩
  simp only [atomicCellCommit, hexecute, bind_ok, hwrites, ite_true, hencode,
    hcontract.controls.width, hwrite]

theorem AtomicWriteSafety.no_stale_metadata {before after : Slice Cell} {destination : Key}
    {encoded : Array U8 8#usize} {resultDefined : Bool} {width : U8}
    (hsafety : AtomicWriteSafety before after destination encoded resultDefined width) :
    ∀ cell ∈ after.val,
      (Overlaps destination width.val cell.key → cell.word = none) ∧
      (Touches destination width.val cell.key →
        cell.origin = none ∧ cell.tainted = true ∧ cell.defined = resultDefined ∧ cell.written = true) := by
  unfold AtomicWriteSafety at hsafety
  generalize before.val = oldCells at hsafety
  generalize after.val = newCells at hsafety ⊢
  induction hsafety with
  | nil => simp
  | @cons old next olds nexts hcell hrest ih =>
    intro cell hmem
    rcases List.mem_cons.mp hmem with rfl | hmem
    · constructor
      · intro hoverlap
        have hoverlap' : Overlaps destination width.val old.key := by
          simpa only [hcell.key] using hoverlap
        simp only [hcell.history, hoverlap', ite_true]
      · intro htouched
        have htouched' : Touches destination width.val old.key := by
          simpa only [hcell.key] using htouched
        have hbytes := hcell.bytes
        simp only [htouched', ite_true] at hbytes
        exact hbytes.2
    · exact ih cell hmem

theorem rust_failed_cas_cell_commit {F : Type}
    (implementation : atomic_exec.FloatingArithmetic F) (input : atomic_exec.AtomicInput)
    (resultDefined : Bool) (before : Slice Cell) (destination : Key)
    (hoperation : input.op = .Integer .Cas) (hwidth : input.bytes = 4#u8 ∨ input.bytes = 8#u8)
    (hcompare : integerNormalize input.bytes input.old ≠ integerNormalize input.bytes input.compare) :
    ∃ outcome after,
      atomic_exec.execute implementation input = .ok (.Ok outcome) ∧
      AtomicCompareExchangeContract input outcome ∧
      outcome.did_write = false ∧ outcome.advance_epoch = false ∧
      outcome.returned = integerNormalize input.bytes input.old ∧ outcome.predicate = true ∧
      words.install_word before ⟨destination, none⟩ = .ok after ∧
      atomicCellCommit implementation input resultDefined before destination = .ok (.Ok (outcome, after)) ∧
      after.val = before.val.map (fun cell => wordPatchedCell cell ⟨destination, none⟩) ∧
      after.val.map numericByteView = before.val.map numericByteView := by
  obtain ⟨outcome, hexecute, hcontract⟩ := rust_atomic_compare_exchange_progress implementation
    input .Cas hoperation (Or.inl rfl) hwidth
  have hswapped : outcome.swapped = false := by simpa [hcompare] using hcontract.swapped
  have hwrites : outcome.did_write = false := by
    simpa [hoperation, atomicIsCas, hswapped] using hcontract.controls.writes
  have hepoch : outcome.advance_epoch = false := by
    simpa only [hwrites] using hcontract.controls.epoch
  have hreturned : outcome.returned = integerNormalize input.bytes input.old := by
    simpa [hoperation, atomicIsSpin, hcontract.old] using hcontract.controls.returned
  have hpredicate : outcome.predicate = true := by
    simpa [hoperation, atomicIsSpin] using hcontract.controls.predicate
  obtain ⟨after, hinstall, hmap⟩ := spec_returns _ _
    (rust_install_word_spec before ⟨destination, none⟩)
  refine ⟨outcome, after, hexecute, hcontract, hwrites, hepoch, hreturned, hpredicate,
    hinstall, ?_, hmap, ?_⟩
  · simp only [atomicCellCommit, hexecute, bind_ok, hwrites, Bool.false_eq_true, ite_false,
      hinstall]
  · simp only [hmap, List.map_map, Function.comp_def, numericByteView, wordPatchedCell]
    rfl

theorem rust_failed_cas_memory_frame {F Input : Type}
    (implementation : atomic_exec.FloatingArithmetic F) (input : atomic_exec.AtomicInput)
    (hoperation : input.op = .Integer .Cas) (hwidth : input.bytes = 4#u8 ∨ input.bytes = 8#u8)
    (hcompare : integerNormalize input.bytes input.old ≠ integerNormalize input.bytes input.compare)
    (sem : Semantics Input) (memory : Memory Input) (before : Storage)
    (state : engine.Engine) (key : Key) (hcells : state.cells.slice = before.toCells)
    (hcapacity : state.track_numeric = true → key.space = 1#u8 →
      state.numeric_written.val.length < Usize.max)
    (hstate : Storage.Represents sem memory before) :
    ∃ outcome next,
      atomic_exec.execute implementation input = .ok (.Ok outcome) ∧
      AtomicCompareExchangeContract input outcome ∧ outcome.did_write = false ∧
      outcome.advance_epoch = false ∧ outcome.returned = integerNormalize input.bytes input.old ∧
      numeric.forget_numeric state key = .ok next ∧ RustNumericRefines before state key next ∧
      Represents sem memory next.cells.slice ∧
      next.cells.slice.val.map numericByteView = state.cells.slice.val.map numericByteView ∧
      next.half_written = state.half_written := by
  obtain ⟨outcome, next, hexecute, hcontract, hwrites, hforget, hrefines, hrep, hbytes, hhalf⟩ :=
    rust_failed_cas_numeric_frame implementation input hoperation hwidth hcompare
      sem memory before state key hcells hcapacity hstate
  have hepoch : outcome.advance_epoch = false := by
    simpa only [hwrites] using hcontract.controls.epoch
  have hreturned : outcome.returned = integerNormalize input.bytes input.old := by
    simpa [hoperation, atomicIsSpin, hcontract.old] using hcontract.controls.returned
  exact ⟨outcome, next, hexecute, hcontract, hwrites, hepoch, hreturned,
    hforget, hrefines, hrep, hbytes, hhalf⟩

theorem AtomicWriteSafety.singleton {before after : Slice Cell} {destination : Key}
    {encoded : Array U8 8#usize} {resultDefined : Bool} {width : U8} {old : Cell}
    (hsafety : AtomicWriteSafety before after destination encoded resultDefined width)
    (hbefore : before.val = [old]) :
    ∃ next, after.val = [next] ∧ AtomicWrittenCell old next destination encoded resultDefined width := by
  unfold AtomicWriteSafety at hsafety
  rw [hbefore] at hsafety
  generalize after.val = nextCells at hsafety ⊢
  cases hsafety with
  | cons hcell htail =>
    cases htail
    exact ⟨_, rfl, hcell⟩

theorem AtomicWrittenCell.partial_overlap {before after : Cell} {destination : Key}
    {encoded : Array U8 8#usize} {resultDefined : Bool} {width : U8}
    (hsafety : AtomicWrittenCell before after destination encoded resultDefined width)
    (hoverlap : Overlaps destination width.val before.key)
    (huntouched : ¬Touches destination width.val before.key) :
    after.word = none ∧ numericByteView after = numericByteView before := by
  constructor
  · simpa only [hoverlap, ite_true] using hsafety.history
  · have hbytes := hsafety.bytes
    simp only [huntouched, ite_false] at hbytes
    rcases hbytes with ⟨hvalue, horigin, htaint, hdefined, hwritten⟩
    simp only [numericByteView, hsafety.key, hvalue, horigin, htaint, hdefined, hwritten]

def staleUpperDestination : Key := ⟨1#u8, 0#u64, 0#u64⟩

def staleUpperCell : Cell :=
  ⟨⟨1#u8, 0#u64, 4#u64⟩, 0#u8, some .Zero, false, true, false, some (.Known 9#u32)⟩

def staleUpperBefore : Slice Cell :=
  Slice.from [staleUpperCell] (by
    have hbounds := Usize.bounds_eq
    simp only [U32.max_eq, U64.max_eq] at hbounds
    simp only [List.length_cons, List.length_nil]
    omega)

def staleUpperCas : atomic_exec.AtomicInput :=
  ⟨.Integer .Cas, 8#u8, 0#u64, 1#u64, 0#u64⟩

theorem rust_cas64_rejects_stale_upper_metadata {F : Type}
    (implementation : atomic_exec.FloatingArithmetic F) :
    ∃ outcome encoded after upper,
      atomic_exec.execute implementation staleUpperCas = .ok (.Ok outcome) ∧
      atomic_bytes.encode outcome.replacement = .ok encoded ∧
      atomicCellCommit implementation staleUpperCas false staleUpperBefore staleUpperDestination =
        .ok (.Ok (outcome, after)) ∧
      after.val = [upper] ∧ upper.word = none ∧ upper.origin = none ∧
      upper.tainted = true ∧ upper.defined = false ∧ upper.written = true ∧
      after.val ≠ staleUpperBefore.val := by
  obtain ⟨outcome, encoded, after, hexecute, _, _, hencode, hcommit, _, hsafety⟩ :=
    rust_successful_cas_cell_commit implementation staleUpperCas false staleUpperBefore
      staleUpperDestination rfl (Or.inr rfl) rfl
  obtain ⟨upper, hlist, hcell⟩ := hsafety.singleton rfl
  have hhistory : upper.word = none := by
    simpa [staleUpperCas, staleUpperDestination, staleUpperCell, Overlaps, SameRegion]
      using hcell.history
  have hbytes : upper.origin = none ∧ upper.tainted = true ∧
      upper.defined = false ∧ upper.written = true := by
    have hbytes := hcell.bytes
    simp [staleUpperCas, staleUpperDestination, staleUpperCell, Touches, SameRegion] at hbytes
    exact hbytes.2
  refine ⟨outcome, encoded, after, upper, hexecute, hencode, hcommit, hlist,
    hhistory, hbytes.1, hbytes.2.1, hbytes.2.2.1, hbytes.2.2.2, ?_⟩
  intro hequal
  have hupper : upper = staleUpperCell := by
    simpa only [hlist, staleUpperBefore, Slice.from_val, List.cons.injEq, and_true] using hequal
  simp only [hupper, staleUpperCell] at hhistory
  cases hhistory

end MemoryWrite
