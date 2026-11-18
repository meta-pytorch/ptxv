/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.PredicateRead

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false

namespace Instruction.PredicateRead

open RegisterOwner

def KnowledgeSound (known : Reg → Option Bool) (actual : Reg → Bool) : Prop :=
  ∀ register value, known register = some value → value = actual register

theorem meaning_conservative (front : Front) (predicate : Pred)
    (known : Reg → Option Bool) (actual : Reg → Bool) (sound : KnowledgeSound known actual)
    (value : Bool) (returned : meaning front predicate known = some value) :
    value = predicateRead front actual predicate := by
  rcases predicate with ⟨register, neg⟩
  cases register with
  | none => simpa [meaning, predicateRead] using returned.symm
  | some register =>
      by_cases constant : immutablePredicate front register
      · simpa [meaning, predicateRead, constant] using returned.symm
      · cases found : known register with
        | none => simp [meaning, constant, found] at returned
        | some observed =>
            have correct := sound register observed found
            simpa [meaning, predicateRead, constant, found, correct] using returned.symm

@[step]
theorem rust_evaluate_conservative {R : Type} (reader : predicate_read.KnownPredicate R)
    (state : R) (front : Front) (predicate : Pred) (known : Reg → Option Bool)
    (actual : Reg → Bool) (sound : KnowledgeSound known actual)
    (frame : R → Prop) (unchanged : frame state)
    (reads : ∀ register, (normalizedPredicate front predicate).reg = some register →
      reader.read state register ⦃ result => result.1 = known register ∧ frame result.2 ⦄) :
    predicate_read.evaluate reader state front predicate ⦃ result =>
      (∀ value, result.1 = some value → value = predicateRead front actual predicate) ∧
      frame result.2 ⦄ := by
  apply WP.spec_mono (rust_evaluate reader state front predicate known frame unchanged reads)
  rintro result ⟨observed, framed⟩
  refine ⟨?_, framed⟩
  intro value present
  exact meaning_conservative front predicate known actual sound value (observed.symm.trans present)

end Instruction.PredicateRead
