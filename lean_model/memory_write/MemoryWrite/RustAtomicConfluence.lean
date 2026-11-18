/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AtomicConfluence
import MemoryWrite.Code.Funs

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite.AtomicConfluence

@[step] theorem rust_contains (values : Slice U32) (value : U32) :
    atomic_confluence.contains values value ⦃ result => result = true → value ∈ values.val ⦄ := by
  unfold atomic_confluence.contains atomic_confluence.contains_loop
  apply loop.spec_decr_nat (fun index : Usize => values.val.length - index.val)
    (fun index : Usize => index.val ≤ values.val.length)
  · intro index bounds
    unfold atomic_confluence.contains_loop.body
    by_cases inside : index.val < values.val.length
    · have test : index < Slice.len values := by scalar_tac
      simp only [test, ite_true]
      step with Slice.index_usize_spec values index inside as ⟨item, itemEq⟩
      split
      · rename_i equal
        simp only [WP.spec_ok]
        intro _
        rw [←equal, itemEq]
        exact List.getElem_mem inside
      · step as ⟨next, nextEq⟩
        exact ⟨by scalar_tac, by scalar_tac⟩
    · have test : ¬ index < Slice.len values := by scalar_tac
      simp [test]
  · scalar_tac

variable {F : Type} (implementation : atomic_exec.FloatingArithmetic F)
    (add : U32 → U32 → U32)
    (arithmetic : ∀ old value, implementation.add_f32_ftz old value
      ⦃ result => result = add old value ⦄)

include arithmetic in
@[step] theorem rust_commutes_row (values : Slice U32) (state first : U32) :
    atomic_confluence.commutes_row implementation values state first ⦃ result =>
      result = true → ∀ second ∈ values.val,
        add (add state first) second = add (add state second) first ⦄ := by
  unfold atomic_confluence.commutes_row
  step with arithmetic state first as ⟨left, leftEq⟩
  subst left
  unfold atomic_confluence.commutes_row_loop
  apply loop.spec_decr_nat (fun index : Usize => values.val.length - index.val)
    (fun index : Usize => index.val ≤ values.val.length ∧
      ∀ position, position < index.val →
        add (add state first) values.val[position]! = add (add state values.val[position]!) first)
  · rintro index ⟨bounds, checked⟩
    unfold atomic_confluence.commutes_row_loop.body
    by_cases inside : index.val < values.val.length
    · have test : index < Slice.len values := by scalar_tac
      simp only [test, ite_true]
      step with Slice.index_usize_spec values index inside as ⟨second, secondEq⟩
      step with arithmetic state second as ⟨right, rightEq⟩
      step with arithmetic (add state first) second as ⟨a, aEq⟩
      step with arithmetic right first as ⟨b, bEq⟩
      split
      · simp only [WP.spec_ok, Bool.false_eq_true, false_implies]
      · rename_i equal
        have same : add (add state first) values.val[index.val]! =
            add (add state values.val[index.val]!) first := by
          rw [←List.Inhabited_getElem_eq_getElem! _ _ inside] at *
          apply UScalar.eq_of_val_eq
          simp_all
        step as ⟨next, nextEq⟩
        refine ⟨by scalar_tac, ?_, by scalar_tac⟩
        intro position earlier
        by_cases old : position < index.val
        · exact checked position old
        · have equalIndex : position = index.val := by scalar_tac
          simpa only [equalIndex] using same
    · have test : ¬ index < Slice.len values := by scalar_tac
      simp only [test, ite_false, WP.spec_ok]
      intro _ second member
      obtain ⟨position, positionInside, atPosition⟩ := List.getElem_of_mem member
      simpa only [←List.Inhabited_getElem_eq_getElem! _ _ positionInside, atPosition] using
        checked position (by omega)
  · exact ⟨by scalar_tac, by intro position earlier; scalar_tac⟩

def scanBody {α : Type} (values : Slice α) (check : α → Result Bool) (index : Usize) :
    Result (ControlFlow Usize Bool) := do
  if index < Slice.len values then
    let item ← Slice.index_usize values index
    let accepted ← check item
    if accepted then
      let next ← index + 1#usize
      ok (.cont next)
    else ok (.done false)
  else ok (.done true)

theorem scan_spec {α : Type} [Inhabited α] (values : Slice α)
    (check : α → Result Bool) (property : α → Prop)
    (checked : ∀ value, check value ⦃ result => result = true → property value ⦄) :
    loop (scanBody values check) 0#usize
      ⦃ result => result = true → ∀ value ∈ values.val, property value ⦄ := by
  apply loop.spec_decr_nat (fun index : Usize => values.val.length - index.val)
    (fun index : Usize => index.val ≤ values.val.length ∧
      ∀ position, position < index.val → property values.val[position]!)
  · rintro index ⟨bounds, prior⟩
    unfold scanBody
    by_cases inside : index.val < values.val.length
    · have test : index < Slice.len values := by scalar_tac
      simp only [test, ite_true]
      step with Slice.index_usize_spec values index inside as ⟨value, valueEq⟩
      step with checked value as ⟨accepted, acceptedProperty⟩
      split
      · rename_i accept
        have current : property values.val[index.val]! := by
          rw [←List.Inhabited_getElem_eq_getElem! _ _ inside, ←valueEq]
          exact acceptedProperty accept
        step as ⟨next, nextEq⟩
        refine ⟨by scalar_tac, ?_, by scalar_tac⟩
        intro position earlier
        by_cases old : position < index.val
        · exact prior position old
        · have equalIndex : position = index.val := by scalar_tac
          simpa only [equalIndex] using current
      · simp only [WP.spec_ok, Bool.false_eq_true, false_implies]
    · have test : ¬ index < Slice.len values := by scalar_tac
      simp only [test, ite_false, WP.spec_ok]
      intro _ value member
      obtain ⟨position, positionInside, atPosition⟩ := List.getElem_of_mem member
      simpa only [←List.Inhabited_getElem_eq_getElem! _ _ positionInside, atPosition] using
        prior position (by omega)
  · exact ⟨by scalar_tac, by intro position earlier; scalar_tac⟩

@[step] theorem rust_subset (values allowed : Slice U32) :
    atomic_confluence.subset values allowed ⦃ result =>
      result = true → ∀ value ∈ values.val, value ∈ allowed.val ⦄ := by
  unfold atomic_confluence.subset atomic_confluence.subset_loop
  have checked := scan_spec values (atomic_confluence.contains allowed)
    (fun value => value ∈ allowed.val) (rust_contains allowed)
  have bodies : atomic_confluence.subset_loop.body values allowed =
      scanBody values (atomic_confluence.contains allowed) := by
    funext index
    simp only [scanBody, atomic_confluence.subset_loop.body]
  rw [bodies]
  exact checked

include arithmetic in
@[step] theorem rust_commutes_state (values : Slice U32) (state : U32) :
    atomic_confluence.commutes_state implementation values state ⦃ result => result = true →
      ∀ first ∈ values.val, ∀ second ∈ values.val,
        add (add state first) second = add (add state second) first ⦄ := by
  unfold atomic_confluence.commutes_state atomic_confluence.commutes_state_loop
  exact scan_spec values (fun first => atomic_confluence.commutes_row implementation values state first)
    (fun first => ∀ second ∈ values.val,
      add (add state first) second = add (add state second) first)
    (rust_commutes_row implementation add arithmetic values state)

include arithmetic in
@[step] theorem rust_commutes_layer (values states : Slice U32) :
    atomic_confluence.commutes_layer implementation values states ⦃ result => result = true →
      ∀ state ∈ states.val, ∀ first ∈ values.val, ∀ second ∈ values.val,
        add (add state first) second = add (add state second) first ⦄ := by
  unfold atomic_confluence.commutes_layer atomic_confluence.commutes_layer_loop
  exact scan_spec states (fun state => atomic_confluence.commutes_state implementation values state)
    (fun state => ∀ first ∈ values.val, ∀ second ∈ values.val,
      add (add state first) second = add (add state second) first)
    (rust_commutes_state implementation add arithmetic values)

include arithmetic in
@[step] theorem rust_extends_row (values next : Slice U32) (state : U32) :
    atomic_confluence.extends_row implementation values next state ⦃ result => result = true →
      ∀ value ∈ values.val, add state value ∈ next.val ⦄ := by
  unfold atomic_confluence.extends_row atomic_confluence.extends_row_loop
  have allChecked := scan_spec values
    (fun value => do
      let after ← implementation.add_f32_ftz state value
      atomic_confluence.contains next after)
    (fun value => add state value ∈ next.val) (by
      intro value
      step with arithmetic state value as ⟨after, afterEq⟩
      step with rust_contains next after as ⟨accepted, checked⟩
      simpa only [afterEq] using checked ‹accepted = true›)
  have bodies : atomic_confluence.extends_row_loop.body implementation values next state =
      scanBody values (fun value => do
        let after ← implementation.add_f32_ftz state value
        atomic_confluence.contains next after) := by
    funext index
    simp only [scanBody, atomic_confluence.extends_row_loop.body, Aeneas.Std.bind_assoc]
  rw [bodies]
  exact allChecked

include arithmetic in
@[step] theorem rust_extends_layer (values current next : Slice U32) :
    atomic_confluence.extends_layer implementation values current next ⦃ result => result = true →
      ∀ state ∈ current.val, ∀ value ∈ values.val, add state value ∈ next.val ⦄ := by
  unfold atomic_confluence.extends_layer atomic_confluence.extends_layer_loop
  exact scan_spec current
    (fun state => atomic_confluence.extends_row implementation values next state)
    (fun state => ∀ value ∈ values.val, add state value ∈ next.val)
    (rust_extends_row implementation add arithmetic values next)

include arithmetic in
@[step] theorem rust_reference (values : Slice U32) (start : U32) :
    atomic_confluence.reference implementation values start
      ⦃ result => result = values.val.foldl add start ⦄ := by
  unfold atomic_confluence.reference atomic_confluence.reference_loop
  apply loop.spec_decr_nat
    (fun state : U32 × Usize => values.val.length - state.2.val)
    (fun state : U32 × Usize => state.2.val ≤ values.val.length ∧
      state.1 = (values.val.take state.2.val).foldl add start)
  · rintro ⟨state, index⟩ ⟨bounds, stateEq⟩
    dsimp only at bounds stateEq
    unfold atomic_confluence.reference_loop.body
    by_cases inside : index.val < values.val.length
    · have test : index < Slice.len values := by scalar_tac
      simp only [test, ite_true]
      step with Slice.index_usize_spec values index inside as ⟨value, valueEq⟩
      step with arithmetic state value as ⟨after, afterEq⟩
      step as ⟨next, nextEq⟩
      refine ⟨by scalar_tac, ?_, by scalar_tac⟩
      simp only [nextEq, List.take_succ_eq_append_getElem inside, List.foldl_append,
        List.foldl_cons, List.foldl_nil]
      simpa only [stateEq, valueEq] using afterEq
    · have test : ¬ index < Slice.len values := by scalar_tac
      simp only [test, ite_false, WP.spec_ok]
      simpa only [List.take_of_length_le (by omega : values.val.length ≤ index.val)] using stateEq
  · exact ⟨by scalar_tac, by simp⟩

def layerValues (layers : Slice (alloc.vec.Vec U32)) : List (List U32) :=
  layers.val.map alloc.vec.Vec.val

theorem layerValues_length (layers : Slice (alloc.vec.Vec U32)) :
    (layerValues layers).length = layers.val.length := by simp [layerValues]

theorem layerValues_at (layers : Slice (alloc.vec.Vec U32)) (depth : Nat)
    (inside : depth < layers.val.length) :
    (layerValues layers)[depth]! = (layers.val[depth]!).val := by
  rw [←List.Inhabited_getElem_eq_getElem! _ _ (by simpa [layerValues] using inside)]
  simp only [layerValues, List.getElem_map]
  rw [List.Inhabited_getElem_eq_getElem! _ _ inside]

structure AuditedPrefix (add : U32 → U32 → U32) (values : List U32)
    (layers : List (List U32)) (budget : U64) (cursor : U64 × Usize) : Prop where
  depth_bound : cursor.2.val ≤ layers.length
  work_bound : cursor.1.val ≤ 4194304 + cursor.2.val * 131072
  within_budget : cursor.1.val ≤ budget.val
  sizes : ∀ depth, depth < cursor.2.val → (layers[depth]!).length ≤ 4096
  commute : ∀ depth, depth < cursor.2.val →
    ∀ state ∈ layers[depth]!, ∀ first ∈ values, ∀ second ∈ values,
      add (add state first) second = add (add state second) first
  closure : ∀ depth, depth < cursor.2.val → depth + 1 < layers.length →
    ∀ state ∈ layers[depth]!, ∀ value ∈ values, add state value ∈ layers[depth + 1]!

include arithmetic in
@[step] theorem rust_check_loop (values alphabet domain : Slice U32)
    (layers : Slice (alloc.vec.Vec U32)) (start want : U32) (budget initialWork : U64)
    (width : values.val.length ≤ 1024) (alphabetBound : alphabet.val.length ≤ 32)
    (shape : layers.val.length = values.val.length - 1)
    (initial : start ∈ (layerValues layers)[0]!)
    (members : ∀ value ∈ values.val, value ∈ alphabet.val)
    (commutation : ∀ state ∈ domain.val, ∀ first ∈ alphabet.val, ∀ second ∈ alphabet.val,
      add (add state first) second = add (add state second) first)
    (initialWorkBound : initialWork.val ≤ 4194304) (initialBudget : initialWork.val ≤ budget.val) :
    atomic_confluence.check_layers implementation values alphabet domain layers start want budget initialWork
      ⦃ result => result = true →
        LayerContract add start values.val (layerValues layers) ∧
        values.val.foldl add start = want ∧
        ∀ states ∈ layerValues layers, states.length ≤ 4096 ⦄ := by
  unfold atomic_confluence.check_layers atomic_confluence.check_layers_loop
  apply loop.spec_decr_nat
    (fun cursor : U64 × Usize => layers.val.length - cursor.2.val)
    (AuditedPrefix add values.val (layerValues layers) budget)
  · rintro ⟨work, depth⟩ audit
    have depthBound : depth.val ≤ layers.val.length := by
      simpa only [layerValues_length] using audit.depth_bound
    have workBound : work.val ≤ 4194304 + depth.val * 131072 := audit.work_bound
    have totalLayers : layers.val.length ≤ 1023 := by omega
    unfold atomic_confluence.check_layers_loop.body
    by_cases inside : depth.val < layers.val.length
    · have test : depth < Slice.len layers := by scalar_tac
      simp only [test, ite_true]
      step with Slice.index_usize_spec layers depth inside as ⟨states, statesEq⟩
      have projected : states.val = (layerValues layers)[depth.val]! := by
        rw [statesEq, List.Inhabited_getElem_eq_getElem! _ _ inside, layerValues_at layers _ inside]
      split
      · simp only [WP.spec_ok, Bool.false_eq_true, false_implies]
      · rename_i capacity
        have stateBound : states.val.length ≤ 4096 := by scalar_tac
        step with UScalar.cast_inBounds_spec .U64 (alloc.vec.Vec.len states) (by scalar_tac)
          as ⟨stateCount, stateCountEq⟩
        step with UScalar.cast_inBounds_spec .U64 (Slice.len alphabet) (by scalar_tac)
          as ⟨valueCount, valueCountEq⟩
        have productBound : stateCount.val * valueCount.val ≤ 131072 := by
          apply (Nat.mul_le_mul (by scalar_tac : stateCount.val ≤ 4096)
            (by scalar_tac : valueCount.val ≤ 32)).trans
          decide
        step with U64.mul_spec (x := stateCount) (y := valueCount) (by scalar_tac)
          as ⟨cost, costEq⟩
        step with U64.add_spec (x := work) (y := cost) (by scalar_tac)
          as ⟨nextWork, nextWorkEq⟩
        split
        · simp only [WP.spec_ok, Bool.false_eq_true, false_implies]
        · rename_i affordable
          have nextBudget : nextWork.val ≤ budget.val := by scalar_tac
          step with rust_subset (alloc.vec.Vec.deref states) domain as ⟨contained, containedSpec⟩
          split
          · rename_i containedTrue
            have currentContained := containedSpec containedTrue
            simp only [alloc.vec.Vec.deref, Slice.from_val] at currentContained
            have currentCommutes : ∀ state ∈ states.val, ∀ first ∈ values.val, ∀ second ∈ values.val,
                add (add state first) second = add (add state second) first := by
              intro state stateMember first firstMember second secondMember
              exact commutation state (currentContained state stateMember) first (members first firstMember)
                second (members second secondMember)
            step as ⟨nextDepth, nextDepthEq⟩
            split
            · rename_i hasNext
              have nextInside : nextDepth.val < layers.val.length := by scalar_tac
              step with Slice.index_usize_spec layers nextDepth nextInside as ⟨following, followingEq⟩
              have nextProjected : following.val = (layerValues layers)[nextDepth.val]! := by
                rw [followingEq, List.Inhabited_getElem_eq_getElem! _ _ nextInside,
                  layerValues_at layers _ nextInside]
              step with rust_extends_layer implementation add arithmetic alphabet
                (alloc.vec.Vec.deref states) (alloc.vec.Vec.deref following)
                as ⟨closed, extendsSpec⟩
              split
              · rename_i extendsTrue
                have alphabetClosure := extendsSpec extendsTrue
                simp only [alloc.vec.Vec.deref, Slice.from_val] at alphabetClosure
                have currentClosure : ∀ state ∈ states.val, ∀ value ∈ values.val,
                    add state value ∈ following.val := by
                  intro state stateMember value valueMember
                  exact alphabetClosure state stateMember value (members value valueMember)
                simp only [WP.spec_ok]
                refine ⟨?_, by scalar_tac⟩
                refine ⟨by simpa only [layerValues_length] using (by scalar_tac : nextDepth.val ≤ layers.val.length),
                  by scalar_tac, nextBudget, ?_, ?_, ?_⟩
                · intro earlier earlierBound
                  by_cases previous : earlier < depth.val
                  · exact audit.sizes earlier previous
                  · have same : earlier = depth.val := by scalar_tac
                    simpa only [same, ←projected] using stateBound
                · intro earlier earlierBound
                  by_cases previous : earlier < depth.val
                  · exact audit.commute earlier previous
                  · have same : earlier = depth.val := by scalar_tac
                    simpa only [same, ←projected] using currentCommutes
                · intro earlier earlierBound laterBound
                  by_cases previous : earlier < depth.val
                  · exact audit.closure earlier previous laterBound
                  · have same : earlier = depth.val := by scalar_tac
                    simpa only [projected, nextProjected, nextDepthEq, same] using currentClosure
              · simp only [WP.spec_ok, Bool.false_eq_true, false_implies]
            · rename_i noNext
              have last : nextDepth.val = layers.val.length := by scalar_tac
              simp only [WP.spec_ok]
              refine ⟨?_, by scalar_tac⟩
              refine ⟨by simpa only [layerValues_length, last] using Nat.le_refl layers.val.length,
                by scalar_tac, nextBudget, ?_, ?_, ?_⟩
              · intro earlier earlierBound
                by_cases previous : earlier < depth.val
                · exact audit.sizes earlier previous
                · have same : earlier = depth.val := by scalar_tac
                  simpa only [same, ←projected] using stateBound
              · intro earlier earlierBound
                by_cases previous : earlier < depth.val
                · exact audit.commute earlier previous
                · have same : earlier = depth.val := by scalar_tac
                  simpa only [same, ←projected] using currentCommutes
              · intro earlier earlierBound laterBound
                exact audit.closure earlier (by simp only [layerValues_length] at laterBound; scalar_tac)
                  laterBound
          · simp only [WP.spec_ok, Bool.false_eq_true, false_implies]
    · have test : ¬depth < Slice.len layers := by scalar_tac
      have finished : depth.val = layers.val.length := by omega
      simp only [test, ite_false]
      step with rust_reference implementation add arithmetic values start as ⟨reference, referenceEq⟩
      intro accepted
      refine ⟨⟨by simpa only [layerValues_length] using shape, by intro _; exact initial,
        ?_, ?_⟩, ?_, ?_⟩
      · intro earlier earlierBound
        exact audit.closure earlier (by change earlier < depth.val; simp only [layerValues_length] at earlierBound; omega)
          earlierBound
      · intro earlier earlierBound
        exact audit.commute earlier (by simpa only [finished, layerValues_length] using earlierBound)
      · have same : reference = want := by simpa using accepted
        exact referenceEq.symm.trans same
      · intro states member
        obtain ⟨earlier, earlierInside, atEarlier⟩ := List.getElem_of_mem member
        have checked := audit.sizes earlier
          (by simpa only [finished, layerValues_length] using earlierInside)
        simpa only [←List.Inhabited_getElem_eq_getElem! _ _ earlierInside, atEarlier] using checked
  · exact ⟨by scalar_tac, by scalar_tac, initialBudget,
      by intro earlier inside; scalar_tac, by intro earlier inside; scalar_tac,
      by intro earlier inside; scalar_tac⟩


include arithmetic in
@[step] theorem rust_check_bounded (values alphabet domain : Slice U32)
    (layers : Slice (alloc.vec.Vec U32)) (start want : U32) (budget : U64)
    (width : values.val.length ≤ 1024) (alphabetBound : alphabet.val.length ≤ 32)
    (domainBound : domain.val.length ≤ 4096) :
    atomic_confluence.check implementation values alphabet domain layers start want budget
      ⦃ result => result = true →
        LayerContract add start values.val (layerValues layers) ∧
        values.val.foldl add start = want ∧
        ∀ states ∈ layerValues layers, states.length ≤ 4096 ⦄ := by
  unfold atomic_confluence.check
  have valuesTest : ¬ Slice.len values > 1024#usize := by scalar_tac
  have alphabetTest : ¬ Slice.len alphabet > 32#usize := by scalar_tac
  have domainTest : ¬ Slice.len domain > 4096#usize := by scalar_tac
  simp only [valuesTest, alphabetTest, domainTest, ite_false]
  by_cases small : values.val.length < 2
  · have test : Slice.len values < 2#usize := by scalar_tac
    simp only [test, ite_true]
    step with core.slice.Slice.is_empty_spec layers as ⟨empty, emptyEq⟩
    split
    · rename_i emptyTrue
      have emptyLayers : layers.val.length = 0 := by scalar_tac
      have projectedEmpty : layerValues layers = [] := by
        apply List.eq_nil_of_length_eq_zero
        simpa only [layerValues_length] using emptyLayers
      step with rust_reference implementation add arithmetic values start as ⟨reference, referenceEq⟩
      intro accepted
      have referenceWant : reference = want := by simpa using accepted
      refine ⟨⟨by simp only [projectedEmpty, List.length_nil]; omega, ?_, ?_, ?_⟩,
        referenceEq.symm.trans referenceWant, ?_⟩
      · intro impossible; omega
      · intro depth inside; simp only [projectedEmpty, List.length_nil] at inside; omega
      · intro depth inside; simp only [projectedEmpty, List.length_nil] at inside; omega
      · simp only [projectedEmpty, List.not_mem_nil, false_implies, implies_true]
    · simp only [WP.spec_ok, Bool.false_eq_true, false_implies]
  · have test : ¬ Slice.len values < 2#usize := by scalar_tac
    simp only [test, ite_false]
    step as ⟨required, requiredEq⟩
    split
    · simp only [WP.spec_ok, Bool.false_eq_true, false_implies]
    · rename_i shapeTest
      have shape : layers.val.length = values.val.length - 1 := by scalar_tac
      have firstInside : 0 < layers.val.length := by omega
      step with Slice.index_usize_spec layers 0#usize firstInside as ⟨first, firstEq⟩
      step with rust_contains (alloc.vec.Vec.deref first) start as ⟨seeded, seededSpec⟩
      split
      · rename_i seededTrue
        have initial : start ∈ (layerValues layers)[0]! := by
          have member := seededSpec seededTrue
          simp only [alloc.vec.Vec.deref, Slice.from_val] at member
          rw [layerValues_at layers 0 firstInside]
          simpa only [firstEq, List.Inhabited_getElem_eq_getElem! _ _ firstInside] using member
        step with rust_subset values alphabet as ⟨admitted, admittedSpec⟩
        split
        · rename_i admittedTrue
          have members := admittedSpec admittedTrue
          step with UScalar.cast_inBounds_spec .U64 (Slice.len domain) (by scalar_tac)
            as ⟨domainCount, domainCountEq⟩
          step with UScalar.cast_inBounds_spec .U64 (Slice.len alphabet) (by scalar_tac)
            as ⟨alphabetCount, alphabetCountEq⟩
          have productBound : domainCount.val * alphabetCount.val ≤ 131072 := by
            apply (Nat.mul_le_mul (by scalar_tac : domainCount.val ≤ 4096)
              (by scalar_tac : alphabetCount.val ≤ 32)).trans
            decide
          step with U64.mul_spec (x := domainCount) (y := alphabetCount) (by scalar_tac)
            as ⟨product, productEq⟩
          step with UScalar.cast_inBounds_spec .U64 (Slice.len alphabet) (by scalar_tac)
            as ⟨again, againEq⟩
          have costBound : product.val * again.val ≤ 4194304 := by
            apply (Nat.mul_le_mul (by scalar_tac : product.val ≤ 131072)
              (by scalar_tac : again.val ≤ 32)).trans
            decide
          step with U64.mul_spec (x := product) (y := again) (by scalar_tac)
            as ⟨work, workEq⟩
          split
          · simp only [WP.spec_ok, Bool.false_eq_true, false_implies]
          · rename_i affordable
            step with rust_commutes_layer implementation add arithmetic alphabet domain
              as ⟨commutes, commutesSpec⟩
            split
            · rename_i commutesTrue
              step with rust_check_loop implementation add arithmetic values alphabet domain layers
                start want budget work width alphabetBound shape initial members
                (commutesSpec commutesTrue) (by scalar_tac) (by scalar_tac)
                as ⟨accepted, checked⟩
              exact checked ‹accepted = true›
            · simp only [WP.spec_ok, Bool.false_eq_true, false_implies]
        · simp only [WP.spec_ok, Bool.false_eq_true, false_implies]
      · simp only [WP.spec_ok, Bool.false_eq_true, false_implies]

include arithmetic in
@[step] theorem rust_check (values alphabet domain : Slice U32)
    (layers : Slice (alloc.vec.Vec U32)) (start want : U32) (budget : U64) :
    atomic_confluence.check implementation values alphabet domain layers start want budget
      ⦃ result => result = true →
        LayerContract add start values.val (layerValues layers) ∧
        values.val.foldl add start = want ∧
        values.val.length ≤ 1024 ∧ alphabet.val.length ≤ 32 ∧ domain.val.length ≤ 4096 ∧
        ∀ states ∈ layerValues layers, states.length ≤ 4096 ⦄ := by
  by_cases width : values.val.length ≤ 1024
  · by_cases alphabetBound : alphabet.val.length ≤ 32
    · by_cases domainBound : domain.val.length ≤ 4096
      · step with rust_check_bounded implementation add arithmetic values alphabet domain layers start want budget
          width alphabetBound domainBound as ⟨accepted, checked⟩
        obtain ⟨contract, reference, sizes⟩ := checked ‹accepted = true›
        exact ⟨contract, reference, width, alphabetBound, domainBound, sizes⟩
      · have valuesTest : ¬ Slice.len values > 1024#usize := by scalar_tac
        have alphabetTest : ¬ Slice.len alphabet > 32#usize := by scalar_tac
        have domainTest : Slice.len domain > 4096#usize := by scalar_tac
        simp [atomic_confluence.check, valuesTest, alphabetTest, domainTest]
    · have valuesTest : ¬ Slice.len values > 1024#usize := by scalar_tac
      have alphabetTest : Slice.len alphabet > 32#usize := by scalar_tac
      simp [atomic_confluence.check, valuesTest, alphabetTest]
  · have valuesTest : Slice.len values > 1024#usize := by scalar_tac
    simp [atomic_confluence.check, valuesTest]
include arithmetic in
@[step] theorem source_all_permutations_match (values alphabet domain : Slice U32)
    (layers : Slice (alloc.vec.Vec U32)) (start want : U32) (budget : U64) :
    atomic_confluence.check implementation values alphabet domain layers start want budget
      ⦃ result => result = true → ∀ order, values.val.Perm order →
        order.foldl add start = want ⦄ := by
  step with rust_check implementation add arithmetic values alphabet domain layers start want budget
    as ⟨accepted, checked⟩
  rename_i acceptedTrue order permutation
  obtain ⟨contract, reference, _, _, _, _⟩ := checked acceptedTrue
  exact (all_permutations_match add start values.val (layerValues layers) contract order permutation).trans
    reference

include arithmetic in
@[step] theorem source_every_admissible_prefix_is_covered (values alphabet domain : Slice U32)
    (layers : Slice (alloc.vec.Vec U32)) (start want : U32) (budget : U64) :
    atomic_confluence.check implementation values alphabet domain layers start want budget
      ⦃ result => result = true → ∀ prior : List U32,
        (∀ value ∈ prior, value ∈ values.val) → prior.length + 2 ≤ values.val.length →
        prior.foldl add start ∈ (layerValues layers)[prior.length]! ∧
        ∀ first ∈ values.val, ∀ second ∈ values.val,
          add (add (prior.foldl add start) first) second =
            add (add (prior.foldl add start) second) first ⦄ := by
  step with rust_check implementation add arithmetic values alphabet domain layers start want budget
    as ⟨accepted, checked⟩
  rename_i acceptedTrue prior members length
  obtain ⟨contract, _, _, _, _, _⟩ := checked acceptedTrue
  have inside : prior.length < (layerValues layers).length := by have := contract.length; omega
  exact ⟨prefix_in_layer add start values.val (layerValues layers) contract prior members inside,
    layers_imply_prefix_commutation add start values.val (layerValues layers) contract prior members length⟩

end MemoryWrite.AtomicConfluence
