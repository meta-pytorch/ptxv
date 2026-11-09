import GB300Math.TensorCore

namespace GB300Math.TensorAlignmentProperties
open GB300Math.Tensor

abbrev Term := GB300Math.Tensor.Dyadic × Int

def selectedScale (terms : List Term) : Int :=
  max (terms.foldl (fun top term => max top term.2) (-10000) - 25) (-158)

def alignedTotal (terms : List Term) : Int :=
  let scale := selectedScale terms
  terms.foldl (fun total term =>
    total + shiftTruncate term.1.significand (term.1.exponent-scale)) 0

def roundedTotal (terms : List Term) : Nat :=
  encodeRz ⟨alignedTotal terms, selectedScale terms⟩

def arrayCore (terms : Array Term) : Nat :=
  let top := terms.foldl (fun s t => max s t.2) (-10000)
  let scale := max (top - 25) (-158)
  let total := terms.foldl
    (fun s t => s + shiftTruncate t.1.significand (t.1.exponent-scale)) 0
  encodeRz ⟨total, scale⟩

theorem arrayCore_eq_list (terms : Array Term) :
    arrayCore terms = roundedTotal terms.toList := by
  simp only [arrayCore, roundedTotal, alignedTotal, selectedScale, Array.foldl_toList]

theorem selectedScale_perm {xs ys : List Term} (h : xs.Perm ys) :
    selectedScale xs = selectedScale ys := by
  have hm := h.foldl_eq' (f := fun top term => max top term.2)
    (by intros; omega) (-10000)
  simp only [selectedScale, hm]

theorem alignedTotal_perm {xs ys : List Term} (h : xs.Perm ys) :
    alignedTotal xs = alignedTotal ys := by
  have hs := selectedScale_perm h
  unfold alignedTotal
  rw [hs]
  exact h.foldl_eq' (by intros; omega) 0

theorem roundedTotal_perm {xs ys : List Term} (h : xs.Perm ys) :
    roundedTotal xs = roundedTotal ys := by
  simp only [roundedTotal, alignedTotal_perm h, selectedScale_perm h]

theorem arrayCore_perm {xs ys : Array Term} (h : xs.toList.Perm ys.toList) :
    arrayCore xs = arrayCore ys := by
  rw [arrayCore_eq_list, arrayCore_eq_list, roundedTotal_perm h]

theorem minimum_alignment_step (terms : List Term) : -158 ≤ selectedScale terms := by
  unfold selectedScale
  omega

theorem unsigned_alignment_loss (n step : Nat) (positive : 0 < step) :
    n / step * step ≤ n ∧ n < n / step * step + step := by
  have h := Nat.mod_add_div n step
  have hr := Nat.mod_lt n positive
  rw [Nat.mul_comm step (n / step)] at h
  omega

#print axioms selectedScale_perm
#print axioms alignedTotal_perm
#print axioms roundedTotal_perm
#print axioms arrayCore_eq_list
#print axioms arrayCore_perm
#print axioms minimum_alignment_step
#print axioms unsigned_alignment_loss

end GB300Math.TensorAlignmentProperties
