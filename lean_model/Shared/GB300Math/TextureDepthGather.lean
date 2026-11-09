import GB300Math.TextureDepthPCF

namespace GB300Math.TextureDepthGather

def pack (a b c d : Bool) : Nat := a.toNat + 2*b.toNat + 4*c.toNat + 8*d.toNat

def sample (depths : Array Nat) (x y mode function reference : Nat) (dx dy : Int) : Nat :=
  let px := TextureDepthPCF.coordinate x / 256 + dx
  let py := TextureDepthPCF.coordinate y / 256 + dy
  let fetch := fun (ix iy : Int) =>
    let depth := match TextureDepthPCF.index ix mode, TextureDepthPCF.index iy mode with
      | some a, some b => depths[b*16+a]!
      | _, _ => 0x3e800000
    TextureDepthCompare.compare function reference depth
  pack (fetch px (py+1)) (fetch (px+1) (py+1)) (fetch (px+1) py) (fetch px py)

theorem pack_bounded (a b c d : Bool) : pack a b c d < 16 := by
  cases a <;> cases b <;> cases c <;> cases d <;> decide

theorem pack_low_bit (a b c d : Bool) : pack a b c d % 2 = a.toNat := by
  cases a <;> cases b <;> cases c <;> cases d <;> decide

theorem pack_injective (a b c d e f g h : Bool)
    (eq : pack a b c d = pack e f g h) : a=e ∧ b=f ∧ c=g ∧ d=h := by
  cases a <;> cases b <;> cases c <;> cases d <;>
    cases e <;> cases f <;> cases g <;> cases h <;> simp_all [pack]

end GB300Math.TextureDepthGather
