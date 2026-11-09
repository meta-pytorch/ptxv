import GB300Math.TextureAnisotropyTaps
namespace GB300Math.TextureAnisotropyTaps
set_option maxRecDepth 100000
set_option maxHeartbeats 0

theorem reciprocal_rn_brackets : ∀ q : Fin 256,
    (2*reciprocalTable[q]!-1)^256*2^q.val ≤ 512^256 ∧
    512^256 < (2*reciprocalTable[q]!+1)^256*2^q.val := by decide

theorem contribution_range (w f : Nat) (vertical : Bool) (a b lo hi : Int)
    (hf : f ≤ 256) (ha : lo ≤ a ∧ a ≤ hi) (hb : lo ≤ b ∧ b ≤ hi) :
    (w:Int)*lo ≤ contribution w f vertical a b ∧
    contribution w f vertical a b ≤ (w:Int)*hi := by
  have hu := upperWeight_bounded w f vertical hf
  have hn : 0 ≤ (w:Int)-(upperWeight w f vertical:Int) := by omega
  have hp : 0 ≤ (upperWeight w f vertical:Int) := Int.natCast_nonneg _
  have l1 := Int.mul_le_mul_of_nonneg_left ha.1 hn
  have l2 := Int.mul_le_mul_of_nonneg_left hb.1 hp
  have h1 := Int.mul_le_mul_of_nonneg_left ha.2 hn
  have h2 := Int.mul_le_mul_of_nonneg_left hb.2 hp
  have clo : ((w:Int)-(upperWeight w f vertical:Int))*lo+
      (upperWeight w f vertical:Int)*lo = (w:Int)*lo := by
    rw [← Int.add_mul]
    congr 1
    omega
  have chi : ((w:Int)-(upperWeight w f vertical:Int))*hi+
      (upperWeight w f vertical:Int)*hi = (w:Int)*hi := by
    rw [← Int.add_mul]
    congr 1
    omega
  unfold contribution
  dsimp only
  constructor <;> omega

theorem symmetric_ideal : ∀ q : Fin 1025, ∀ negative : Bool,
    ((taps q negative).2.map fun t => (t.weight:Int)*t.offset).foldl (·+·) 0 = 0 := by decide

def centerNumerator (q : Nat) (negative vertical : Bool) : Int :=
  ((taps q negative).2.map fun t => contribution t.weight
    (t.offset%256).toNat vertical (t.offset/256) (t.offset/256+1)).foldl (·+·) 0

theorem center_shift_bound : ∀ q : Fin 1025, ∀ negative vertical : Bool,
    (centerNumerator q negative vertical).natAbs*128 ≤ (taps q negative).1*256 := by decide

#print axioms reciprocal_rn_brackets
#print axioms contribution_range
#print axioms symmetric_ideal
#print axioms center_shift_bound
end GB300Math.TextureAnisotropyTaps
