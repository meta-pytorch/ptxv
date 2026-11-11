import GB300Math.TextureGradient2D

namespace GB300Math.TextureGradient3D
open TextureGradient2D

def roundEven (n divisor : Nat) : Nat :=
  let q := n/divisor
  let r := n%divisor
  q + if 2*r > divisor ∨ (2*r = divisor ∧ q%2 = 1) then 1 else 0

def rounded32 (a : TextureGradient2D.Dyadic) : TextureGradient2D.Dyadic :=
  let n := a.significand.natAbs
  if n = 0 then ⟨0,0⟩ else
    let shift := max 0 (max ((n.log2:Int)-23) (-149-a.scale))
    ⟨signed (a.significand < 0) (roundEven n (2^shift.toNat)), a.scale+shift⟩

def exactFromBits (v : Nat) : TextureGradient2D.Dyadic :=
  let e : Nat := v/2^23%256
  let m : Nat := v%2^23
  if e = 0 then ⟨signed (v ≥ 2^31) m,-149⟩ else
    ⟨signed (v ≥ 2^31) (m+2^23), (e:Int)-150⟩

def adjusted (coordinate v : Nat) : TextureGradient2D.Dyadic :=
  let c := exactFromBits coordinate
  if v%2^31 > 0x7f800000 then normalized (negate c) else
    normalized (rounded32 (add (rounded32 (add (exactFromBits v) c)) (negate c)))

def ordered (a b c : TextureGradient2D.Dyadic) : TextureGradient2D.Dyadic × TextureGradient2D.Dyadic × TextureGradient2D.Dyadic :=
  let a := magnitude a
  let b := magnitude b
  let c := magnitude c
  let (a,b) := if le a b then (a,b) else (b,a)
  let (b,c) := if le b c then (b,c) else (c,b)
  let (a,b) := if le a b then (a,b) else (b,a)
  (a,b,c)

def length (a b c : TextureGradient2D.Dyadic) : TextureGradient2D.Dyadic :=
  let (lo,mid,hi) := ordered a b c
  add (add hi (normalized (scaleCoefficient mid 11 5 3))) ⟨lo.significand,lo.scale-2⟩

def diagonal (a b c : TextureGradient2D.Dyadic) : TextureGradient2D.Dyadic :=
  let (lo,mid,hi) := ordered a b c
  add (add (normalized (scaleCoefficient hi 11 4 3))
    (normalized (scaleCoefficient mid 121 9 6))) (scaleCoefficient lo 11 6 3)

def norm (ax ay az bx byy bz : TextureGradient2D.Dyadic) : TextureGradient2D.Dyadic :=
  let p := diagonal (guardedAdd ax bx) (guardedAdd ay byy) (guardedAdd az bz)
  let m := diagonal (guardedAdd ax (negate bx)) (guardedAdd ay (negate byy)) (guardedAdd az (negate bz))
  maximum (maximum (length ax ay az) (length bx byy bz)) (maximum p m)

def position (cx cy cz ax ay az bx byy bz : Nat) : Nat :=
  if [ax,ay,az,bx,byy,bz].any (fun v => v%2^31 = 0x7f800000) then 1280 else
    let r := norm (adjusted cx ax) (adjusted cy ay) (adjusted cz az) (adjusted cx bx) (adjusted cy byy) (adjusted cz bz)
    let n := r.significand.natAbs
    if n = 0 then 0 else
      let top := n.log2
      let index := n*256/2^top%256
      min 1280 ((r.scale+(top:Int)+5)*256+
        (TextureGradientAxis.logTable[index]!:Int)).toNat

def sample (cx cy cz ax ay az bx byy bz : Nat) : Nat := TextureMipmap.encodeQ8 (256+position cx cy cz ax ay az bx byy bz)

theorem position_bounded (cx cy cz ax ay az bx byy bz : Nat) : position cx cy cz ax ay az bx byy bz ≤ 1280 := by
  unfold position
  dsimp only
  split
  · exact Nat.le_refl _
  · split
    · exact Nat.zero_le _
    · exact Nat.min_le_left _ _

#print axioms position_bounded
end GB300Math.TextureGradient3D
