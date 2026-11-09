import Std.Tactic
namespace GB300Math.TextureSeamless
def weights (x y : Nat) : Array Nat :=
  let ll := ((256-x)*(256-y)+128)/256
  let hh := (x*y+128)/256
  #[ll,x-hh,256-x-ll,hh]

def support (x y i : Nat) : Bool :=
  (if i%2 = 0 then x < 256 else x > 0) &&
  (if i/2 = 0 then y < 256 else y > 0)

def magnitude (v : Nat) : Nat := v % 2^31

def exponent (v : Nat) : Nat := v / 2^23 % 256

def negative (v : Nat) : Bool := v ≥ 2^31

def encode (value scale : Int) : Nat := Id.run do
  if value = 0 then return 0
  let sign := if value < 0 then 2^31 else 0
  let n := value.natAbs
  let shift := max ((n.log2 : Int)-23) (-149-scale)
  let rounded := if shift > 0 then
      let divisor := 2^shift.toNat
      n/divisor + if n%divisor ≥ divisor/2 then 1 else 0
    else n*2^(-shift).toNat
  let e := scale+shift
  if rounded = 0 then return sign
  let (m,e) := if rounded ≥ 2^24 then (rounded/2,e+1) else (rounded,e)
  let biased := (m.log2 : Int)+e+127
  if biased ≥ 255 then return sign+0x7f800000
  if biased ≤ 0 then return sign
  return sign+biased.toNat*2^23+m%2^23

def weighted (ws geometric values : Array Nat) : Nat := Id.run do
  if geometric.any (fun i => magnitude values[i]! > 0x7f800000) then return 0x7fffffff
  let infs := geometric.filter (fun i => magnitude values[i]! = 0x7f800000)
  if !infs.isEmpty then
    let sign := negative values[infs[0]!]!
    if infs.any (fun i => negative values[i]! != sign) then return 0x7fffffff
    return (if sign then 2^31 else 0)+0x7f800000
  let finite := geometric.filter (fun i => ws[i]! > 0 && exponent values[i]! > 0)
  if finite.isEmpty then
    return if geometric.all (fun i => negative values[i]! && exponent values[i]! = 0)
      then 2^31 else 0
  let top := finite.foldl (fun e i => max e (exponent values[i]!)) 0
  let scale := (top : Int)-154
  let mut total : Int := 0
  for i in finite do
    let value := values[i]!
    let mantissa := value%2^23+2^23
    let shift := (exponent value : Int)-150-scale
    let aligned : Nat := if shift ≥ 0 then mantissa*2^shift.toNat else mantissa/2^(-shift).toNat
    let signed : Int := if negative value then -(aligned : Int) else aligned
    total := total + signed*(ws[i]! : Int)
  return encode total (scale-8)

def word (bytes : ByteArray) (index : Nat) : Nat :=
  (List.range 4).foldl (fun v i => v + bytes[index*4+i]!.toNat * 2^(8*i)) 0

def dyadic (bits : Nat) : Bool :=
  let e := bits / 2^23 % 256
  let m := bits % 2^23 + if e = 0 then 0 else 2^23
  e ≤ 127 && m % 2^(140-max e 1) = 0

def coordinate (bits : Nat) : Int :=
  let e := bits / 2^23 % 256
  let m := bits % 2^23 + if e = 0 then 0 else 2^23
  let v : Nat := m / 2^(140-max e 1)
  if bits / 2^31 = 1 then -(v : Int) else v

def face (v : Array Int) : Nat :=
  let top := v.foldl (fun m x => max m x.natAbs) 0
  let axis := if v[2]!.natAbs = top then 2 else if v[1]!.natAbs = top then 1 else 0
  2*axis + if v[axis]! < 0 then 1 else 0

def project (v : Array Int) : Int × Int :=
  match face v with
  | 0 => (-v[2]!, -v[1]!)
  | 1 => (v[2]!, -v[1]!)
  | 2 => (v[0]!, v[2]!)
  | 3 => (v[0]!, -v[2]!)
  | 4 => (v[0]!, -v[1]!)
  | _ => (-v[0]!, -v[1]!)

def inverse (f : Nat) (x y : Int) : Array Int :=
  let s := 2*x-7
  let t := 2*y-7
  match f with
  | 0 => #[8,-t,-s]
  | 1 => #[-8,-t,s]
  | 2 => #[s,8,t]
  | 3 => #[s,-8,-t]
  | 4 => #[s,-t,8]
  | _ => #[-s,-t,-8]

def clamp (x : Int) : Int := min 7 (max 0 x)

def remap (f : Nat) (x y : Int) : Nat × Nat × Nat :=
  let v := inverse f x y
  let m : Int := v.foldl (fun a b => max a b.natAbs) 0
  let (s,t) := project v
  (face v,(clamp ((s+m)*4/m)).toNat,(clamp ((t+m)*4/m)).toNat)

def fetch (ts : ByteArray) (layer f : Nat) (x y : Int) (channel : Nat) : Nat :=
  word ts ((((layer*6+f)*8+(clamp y).toNat)*8+(clamp x).toNat)*4+channel)

def mapped (ts : ByteArray) (layer f : Nat) (x y : Int) (channel : Nat) : Nat :=
  let (g,u,v) := remap f x y
  fetch ts layer g u v channel

def outside (x : Int) : Bool := x < 0 || x ≥ 8

def canonical (v : Nat) : Nat :=
  if magnitude v > 0x7f800000 then 0x7fffffff
  else if exponent v = 0 then v/2^31*2^31 else v

def gather (ts : ByteArray) (layer f : Nat) (x y : Int) (channel : Nat) (seamless : Bool) : Nat :=
  if !seamless then canonical (fetch ts layer f x y channel)
  else if outside x && outside y then
    weighted #[86,85,85] #[0,1,2]
      #[fetch ts layer f x y channel,
        mapped ts layer f x (clamp y) channel,
        mapped ts layer f (clamp x) y channel]
  else canonical (mapped ts layer f x y channel)

def linear (ts : ByteArray) (layer f : Nat) (s t : Int) (channel : Nat) (seamless : Bool) : Nat := Id.run do
  let a := (s+1024)*8-1024
  let b := (t+1024)*8-1024
  let x := a/2048
  let y := b/2048
  let wx := (a%2048/8).toNat
  let wy := (b%2048/8).toNat
  let mut ws := weights wx wy
  let mut vs : Array Nat := #[]
  let mut geom := (Array.range 4).filter (support wx wy)
  let mut corner : Option Nat := none
  for i in [:4] do
    let xi := x+(i%2:Nat)
    let yi := y+(i/2:Nat)
    if seamless && outside xi && outside yi then
      corner := some i
      vs := vs.push 0
    else
      vs := vs.push (if seamless then mapped ts layer f xi yi channel
        else fetch ts layer f xi yi channel)
  if let some c := corner then
    let amount := ws[c]!
    let share := (amount+1)/3
    ws := ws.set! c 0
    ws := ws.set! (3-c) (ws[3-c]!+amount-2*share)
    for i in [:4] do
      if i != c && i != 3-c then ws := ws.set! i (ws[i]!+share)
    geom := geom.filter (fun i => i != c)
  return weighted ws geom vs

theorem corner_weights_sum : 86+85+85 = (256:Nat) := by decide
theorem corner_differs_from_thirds : weighted #[86,85,85] #[0,1,2] #[0x3f800000,0,0] = 0x3eac0000 := by decide +kernel
theorem negative_subnormal_flush : canonical 0x80000001 = 0x80000000 := by decide
theorem corner_nan : weighted #[86,85,85] #[0,1,2] #[0x7fa12345,0,0] = 0x7fffffff := by decide +kernel
#print axioms corner_weights_sum
#print axioms corner_differs_from_thirds
#print axioms negative_subnormal_flush
#print axioms corner_nan
end GB300Math.TextureSeamless
