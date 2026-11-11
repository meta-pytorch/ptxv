import FloatLib.Floats.Formats.BinaryInterchange.Configured
import FloatLib.Floats.Formats.BinaryInterchange.DirectedSemantics.Bounds
import FloatLib.Floats.Formats.BinaryInterchange.Reduction.Proof
import Mathlib.Tactic

set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace GB300Math.FabricGeneralBounds
open FloatLib.Floats.Formats.BinaryInterchange

theorem dyadic_enclosure (fmt : FloatFormat) (d : FloatLib.Numerics.Dyadic)
    (hfmt : fmt.isIEEE = true)
    (hdown : Model.isFinite (Model.roundDyadicDown fmt d) = true)
    (hup : Model.isFinite (Model.roundDyadicUp fmt d) = true) :
    Model.toReal (Model.roundDyadicDown fmt d) ≤ d.toReal ∧
    d.toReal ≤ Model.toReal (Model.roundDyadicUp fmt d) := by
  have hd := Model.toEReal_roundDyadicDown_le fmt d hfmt
  have hu := Model.le_toEReal_roundDyadicUp fmt d hfmt
  rw [Model.toEReal_eq_coe_toReal_of_isFinite _ hdown] at hd
  rw [Model.toEReal_eq_coe_toReal_of_isFinite _ hup] at hu
  exact ⟨EReal.coe_le_coe_iff.mp hd, EReal.coe_le_coe_iff.mp hu⟩

theorem dyadic_choice_error (fmt : FloatFormat) (d : FloatLib.Numerics.Dyadic)
    (hfmt : fmt.isIEEE = true)
    (hdown : Model.isFinite (Model.roundDyadicDown fmt d) = true)
    (hup : Model.isFinite (Model.roundDyadicUp fmt d) = true)
    (actual : Model fmt)
    (hactual : actual = Model.roundDyadicDown fmt d ∨ actual = Model.roundDyadicUp fmt d) :
    |Model.toReal actual - d.toReal| ≤
      Model.toReal (Model.roundDyadicUp fmt d) - Model.toReal (Model.roundDyadicDown fmt d) := by
  obtain ⟨hd,hu⟩ := dyadic_enclosure fmt d hfmt hdown hup
  rcases hactual with h | h
  · rw [h, abs_of_nonpos (sub_nonpos.mpr hd)]
    linarith
  · rw [h, abs_of_nonneg (sub_nonneg.mpr hu)]
    linarith

theorem sum_choice_error (fmt source : FloatFormat) (values : Array (Model source))
    (hfmt : fmt.isIEEE = true)
    (hfinite : ∀ value ∈ values.toList, Model.isFinite value = true)
    (hnonzero : Model.Reduction.Semantics.finiteSum values.toList ≠ 0)
    (hdown : Model.isFinite (Model.sum fmt values .towardNegativeInfinity) = true)
    (hup : Model.isFinite (Model.sum fmt values .towardPositiveInfinity) = true)
    (actual : Model fmt)
    (hactual : actual = Model.sum fmt values .towardNegativeInfinity ∨
      actual = Model.sum fmt values .towardPositiveInfinity) :
    |Model.toReal actual - (Model.Reduction.Semantics.finiteSum values.toList : ℝ)| ≤
      Model.toReal (Model.sum fmt values .towardPositiveInfinity) -
      Model.toReal (Model.sum fmt values .towardNegativeInfinity) := by
  let d := (values.foldl Model.Reduction.Internal.State.pushValue
    ({} : Model.Reduction.Internal.State fmt)).exact
  have hd : Model.sum fmt values .towardNegativeInfinity = Model.roundDyadicDown fmt d := by
    unfold Model.sum
    rw [Model.Reduction.sumWithStatus_eq_round_of_finite_nonzero fmt values _ hfinite hnonzero]
    rfl
  have hu : Model.sum fmt values .towardPositiveInfinity = Model.roundDyadicUp fmt d := by
    unfold Model.sum
    rw [Model.Reduction.sumWithStatus_eq_round_of_finite_nonzero fmt values _ hfinite hnonzero]
    rfl
  have hv : d.toReal = (Model.Reduction.Semantics.finiteSum values.toList : ℝ) := by
    rw [← FloatLib.Numerics.Dyadic.cast_toRat]
    congr 1
    exact Model.Reduction.Internal.array_foldl_pushValue_exact_toRat (destination := fmt) values
  rw [hd] at hdown
  rw [hu] at hup
  rw [hd,hu] at hactual ⊢
  rw [← hv]
  exact dyadic_choice_error fmt d hfmt hdown hup actual hactual

#print axioms dyadic_enclosure
#print axioms dyadic_choice_error
#print axioms sum_choice_error
end GB300Math.FabricGeneralBounds
