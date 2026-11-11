import GB300Math.TextureDepthCompare

namespace GB300Math.TextureDepthPCF

def coordinate (value : Nat) : Int :=
  let e : Nat := value / 2^23 % 256
  if e = 0 ∨ e = 255 then -128 else
  let m : Nat := value % 2^23 + 2^23
  let scale := (e : Int) - 138
  let signed : Int := if value ≥ 2^31 then -(m : Int) else m
  (if scale ≥ 0 then signed * 2^scale.toNat
    else (signed + 2^((-scale).toNat-1)) / 2^(-scale).toNat) - 128

def position (value mode : Nat) : Int :=
  let p := coordinate value
  if mode = 1 then max 0 (min 3840 p) else p

def index (i : Int) (mode : Nat) : Option Nat :=
  if mode = 0 then some (i % 16).toNat
  else if mode = 1 then some (min 15 i.toNat)
  else if mode = 2 then
    let r := (i % 32).toNat
    some (if r < 16 then r else 31-r)
  else if i < 0 ∨ i ≥ 16 then none else some i.toNat

def plane (x y : Nat) : Array Nat :=
  let lowLow := ((256-x)*(256-y)+128)/256
  let highHigh := (x*y+128)/256
  #[lowLow, x-highHigh, 256-x-lowLow, highHigh]

def blend (x y : Nat) (a b c d : Bool) : Nat :=
  let w := plane x y
  w[0]! * a.toNat + w[1]! * b.toNat + w[2]! * c.toNat + w[3]! * d.toNat

def idealNumerator (x y : Nat) (a b c d : Bool) : Int :=
  let lo : Int := (256-x)*(256-y)
  let hi : Int := x*y
  lo*a.toNat + ((x*256 : Nat) - hi)*b.toNat +
    (((256-x)*256 : Nat) - lo)*c.toNat + hi*d.toNat

def encode (n : Nat) : Nat :=
  if n = 0 then 0 else (n.log2+119)*2^23 + (n*2^(23-n.log2))%2^23

def sample (depths : Array Nat) (x y mode function reference : Nat) : Nat :=
  let px := position x mode
  let py := position y mode
  let fetch := fun (ix iy : Int) =>
    let depth := match index ix mode, index iy mode with
      | some a, some b => depths[b*16+a]!
      | _, _ => 0x3e800000
    TextureDepthCompare.compare function reference depth
  encode (blend (px%256).toNat (py%256).toNat
    (fetch (px/256) (py/256)) (fetch (px/256+1) (py/256))
    (fetch (px/256) (py/256+1)) (fetch (px/256+1) (py/256+1)))

theorem plane_sum (x y : Nat) (hx : x ≤ 256) (hy : y ≤ 256) :
    (plane x y).foldl (·+·) 0 = 256 := by
  have h1 := Nat.mul_le_mul_left (256-x) (Nat.sub_le 256 y)
  have h2 := Nat.mul_le_mul_left x hy
  simp only [Nat.mul_comm _ 256] at h1 h2
  change 0 + ((256-x)*(256-y)+128)/256 + (x-(x*y+128)/256) +
    (256-x-((256-x)*(256-y)+128)/256) + (x*y+128)/256 = 256
  omega

theorem blend_bounded (x y : Nat) (hx : x ≤ 256) (hy : y ≤ 256)
    (a b c d : Bool) : blend x y a b c d ≤ 256 := by
  have h := plane_sum x y hx hy
  change 0 + ((256-x)*(256-y)+128)/256 + (x-(x*y+128)/256) +
    (256-x-((256-x)*(256-y)+128)/256) + (x*y+128)/256 = 256 at h
  cases a <;> cases b <;> cases c <;> cases d <;>
    simp [blend, plane, Bool.toNat] <;> omega

theorem subnormal_coordinates_flush : coordinate 1 = -128 ∧ coordinate 0x80000001 = -128 := by decide +kernel

theorem negative_halfway_up : coordinate 0xbf008000 = -2184 := by decide +kernel

theorem quarter_blend : blend 64 64 true false false false = 144 := by decide +kernel

end GB300Math.TextureDepthPCF
