import GB300Math.Mufu

namespace GB300Math.MufuTableBounds

private theorem index_of_capacity (index divisor size : Nat)
    (h : index < size * divisor) : index / divisor < size := by
  by_cases hd : divisor = 0
  · simp [hd] at h
  · exact (Nat.div_lt_iff_lt_mul (by omega)).mpr h

theorem scalar_capacities :
    MufuTables.ex2.n = 17 ∧ MufuTables.ex2.coefficients.size = 64 ∧
    MufuTables.rcp.n = 16 ∧ MufuTables.rcp.coefficients.size = 128 ∧
    MufuTables.rsq.n = 17 ∧ MufuTables.rsq.coefficients.size = 128 ∧
    MufuTables.sqrt.n = 17 ∧ MufuTables.sqrt.coefficients.size = 128 ∧
    MufuTables.lg2.n = 17 ∧ MufuTables.lg2.coefficients.size = 64 ∧
    MufuTables.sin.n = 17 ∧ MufuTables.sin.coefficients.size = 64 := by
  decide +kernel

theorem fraction_indices (x : Nat) :
    Mufu.fraction x / 2 ^ MufuTables.rcp.n < MufuTables.rcp.coefficients.size ∧
    Mufu.fraction x / 2 ^ MufuTables.lg2.n < MufuTables.lg2.coefficients.size := by
  have hf := Nat.mod_lt x (by decide : 0 < 2^23)
  simp only [Mufu.fraction]
  rcases scalar_capacities with ⟨_, _, rn, rs, _, _, _, _, ln, ls, _, _⟩
  rw [rn, rs, ln, ls]
  constructor <;> omega

theorem fraction_domain_indices (index : Nat) (h : index < 2^23) :
    index / 2 ^ MufuTables.ex2.n < MufuTables.ex2.coefficients.size ∧
    index / 2 ^ MufuTables.sin.n < MufuTables.sin.coefficients.size := by
  rcases scalar_capacities with ⟨en, es, _, _, _, _, _, _, _, _, sn, ss⟩
  rw [en, es, sn, ss]
  constructor <;> omega

theorem root_index_domain (x : Nat) : (Mufu.rootIndex x).1 < 2^24 := by
  have hf := Nat.mod_lt x (by decide : 0 < 2^23)
  have hp := Int.emod_nonneg ((Mufu.exponent x : Int) - 127) (by decide : (2 : Int) ≠ 0)
  have hq := Int.emod_lt_of_pos ((Mufu.exponent x : Int) - 127) (by decide : (0 : Int) < 2)
  simp only [Mufu.rootIndex, Mufu.fraction]
  omega

theorem root_indices (x : Nat) :
    (Mufu.rootIndex x).1 / 2 ^ MufuTables.rsq.n < MufuTables.rsq.coefficients.size ∧
    (Mufu.rootIndex x).1 / 2 ^ MufuTables.sqrt.n < MufuTables.sqrt.coefficients.size := by
  rcases scalar_capacities with ⟨_, _, _, _, rn, rs, sn, ss, _, _, _, _⟩
  rw [rn, rs, sn, ss]
  have h := root_index_domain x
  constructor <;> omega

theorem tanh_capacities : TanhTables.binades.size = 11 ∧
    ∀ i : Fin 11, 2^23 ≤ (TanhTables.binades[i.val]!).2.size *
      (2 ^ (TanhTables.binades[i.val]!).1 * 2^17) := by
  decide +kernel

theorem tanh_binade_index (e : Nat) (lower : 119 ≤ e) (upper : e < 130) :
    e - 119 < TanhTables.binades.size := by
  rw [tanh_capacities.1]
  omega

theorem tanh_coefficient_index (i : Fin 11) (fraction : Nat) (h : fraction < 2^23) :
    fraction / 2 ^ (TanhTables.binades[i.val]!).1 / 2^17 <
      (TanhTables.binades[i.val]!).2.size := by
  rw [Nat.div_div_eq_div_mul]
  exact index_of_capacity _ _ _ (Nat.lt_of_lt_of_le h (tanh_capacities.2 i))

theorem seed_capacities : SeedTables.rcp.size = 128 ∧ SeedTables.rsq.size = 128 := by
  decide +kernel

theorem seed_indices (fraction : Nat) (parity : Nat)
    (hf : fraction < 2^20) (hp : parity < 2) :
    fraction * 8 / 2^16 < SeedTables.rcp.size ∧
    (parity * 2^23 + fraction * 8) / 2^17 < SeedTables.rsq.size := by
  rw [seed_capacities.1, seed_capacities.2]
  constructor <;> omega

#print axioms scalar_capacities
#print axioms fraction_indices
#print axioms fraction_domain_indices
#print axioms root_index_domain
#print axioms root_indices
#print axioms tanh_capacities
#print axioms tanh_binade_index
#print axioms tanh_coefficient_index
#print axioms seed_capacities
#print axioms seed_indices

end GB300Math.MufuTableBounds
