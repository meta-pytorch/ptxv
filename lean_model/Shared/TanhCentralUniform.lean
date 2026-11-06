import TanhCentralModel
import TanhNativeCore
import FixedNormal
import Exp2Decode
import TanhPlateau
namespace GB300Math.TanhCentralUniform
open TanhCentralCells TanhCentralInteger TanhCentralIndex TanhCentralModel
open FloatLib.Floats.Formats.BinaryInterchange
set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

def inputWord (e:Fin 11) (f:Fin (2^23)) : Nat := (119+e.val)*2^23+f.val

theorem input_decode (e:Fin 11) (f:Fin (2^23)) :
    (Model.ofNatBits (inputWord e f):Model FloatFormat.binary32).toReal=
      (2^23+(f.val:ℝ))*(2:ℝ)^((e.val:Int)-31) := by
  let ei:Fin 254:=⟨118+e.val,by have := e.isLt;omega⟩
  have hd:=MufuRcp32Bounds.decode_normal ei f
  have he : 1+ei.val=119+e.val := by dsimp [ei];omega
  have hp : (ei.val:Int)-149=(e.val:Int)-31 := by dsimp [ei];omega
  rw [he,hp] at hd
  have hr:=Model.Algebraic.toReal_of_toRat?_eq_some hd
  norm_num [inputWord] at hr ⊢
  exact hr

theorem input_fields (e:Fin 11) (f:Fin (2^23)) :
    Mufu.exponent (inputWord e f)=119+e.val ∧ Mufu.fraction (inputWord e f)=f.val ∧
    Mufu.magnitude (inputWord e f)<0x7f800000 ∧ inputWord e f<2^31 := by
  have he:=e.isLt
  have hf:=f.isLt
  unfold inputWord Mufu.exponent Mufu.fraction Mufu.magnitude
  omega

theorem core_value (e f:Nat) : TanhNativeCore.value e f=value e f := by
  unfold TanhNativeCore.value value coeff argument index TanhCentralIndex.discard
  rcases ht : TanhTables.binades[e]! with ⟨d,cs⟩
  dsimp only [Prod.fst,Prod.snd]

theorem native_bits (e:Fin 11) (f:Fin (2^23)) :
    Mufu.tanh (inputWord e f)=some (Mufu.fixedToWord (value e.val f.val).toNat (scale e.val)) := by
  have hh:=input_fields e f
  have hs:=RcpSigned.sign_zero (inputWord e f) hh.2.2.2
  have he:=e.isLt
  have hlo : 119≤Mufu.exponent (inputWord e f) := by rw [hh.1];omega
  have hhi : Mufu.exponent (inputWord e f)<130 := by rw [hh.1];omega
  rw [TanhNativeCore.central (inputWord e f) hh.2.2.1.le hlo hhi]
  rw [hs,hh.1,hh.2.1,Nat.zero_add]
  have hi : 119+e.val-119=e.val := by omega
  rw [hi,core_value]
  have hscale : (scale e.val:Int)=184-(min ((119+e.val:Nat):Int) 126) := by unfold scale;omega
  rw [hscale]

theorem decoded_error (e:Fin 11) (f:Fin (2^23)) :
    ∃ out:Nat,Mufu.tanh (inputWord e f)=some out ∧
      (Model.ofNatBits out:Model FloatFormat.binary32).isFinite=true ∧
      |(Model.ofNatBits out:Model FloatFormat.binary32).toReal-
        Real.tanh ((Model.ofNatBits (inputWord e f):Model FloatFormat.binary32).toReal)| ≤ 1/2^16 := by
  have hv:=value_range e f.val f.isLt
  have hs:58 ≤ scale e.val ∧ scale e.val ≤ 65 := (parameters e).2
  have hnat : 0<(value e.val f.val).toNat ∧ (value e.val f.val).toNat<2^(scale e.val) := by
    constructor
    · omega
    · exact (Int.toNat_lt' (by positivity)).mpr (by exact_mod_cast hv.2)
  obtain ⟨y,hy,hr⟩:=FixedNormal.fixed_subunit (value e.val f.val).toNat (scale e.val) hs hnat
  refine ⟨_,native_bits e f,Model.Algebraic.isFinite_of_toRat?_eq_some hy,?_⟩
  rw [Model.Algebraic.toReal_of_toRat?_eq_some hy,input_decode]
  have he:=unrounded_error e f.val f.isLt
  have hrreal : 0 ≤ (value e.val f.val:ℝ)/2^(scale e.val)-(y:ℝ) ∧
      (value e.val f.val:ℝ)/2^(scale e.val)-(y:ℝ) ≤ 1/2^24 := by
    have h0:=(Rat.cast_nonneg (K:=ℝ)).mpr hr.1
    have h1:=(Rat.cast_le (K:=ℝ)).mpr hr.2
    norm_num only [Rat.cast_sub,Rat.cast_div,Rat.cast_natCast,Rat.cast_pow,Rat.cast_ofNat,Rat.cast_one] at h0 h1
    have hn : ((value e.val f.val).toNat:ℝ)=(value e.val f.val:ℝ) := by
      have h:=Int.toNat_of_nonneg hv.1.le
      exact_mod_cast h
    rw [hn] at h0 h1
    exact ⟨h0,by norm_num at h1 ⊢;exact h1⟩
  have hround : |(y:ℝ)-(value e.val f.val:ℝ)/2^(scale e.val)| ≤ 1/2^24 := by
    rw [abs_sub_comm,abs_of_nonneg hrreal.1]
    exact hrreal.2
  have ht:=abs_sub_le (y:ℝ) ((value e.val f.val:ℝ)/2^(scale e.val))
    (Real.tanh ((2^23+(f.val:ℝ))*(2:ℝ)^((e.val:Int)-31)))
  norm_num only [Nat.reducePow] at hround ht ⊢
  linarith only [ht,hround,he]

theorem actual_value (word:Nat) (hw:word<2^32) (hf:Mufu.magnitude word<0x7f800000) :
    (Model.ofNatBits word:Model FloatFormat.binary32).toReal=Exp2Bounds.finiteValue word := by
  obtain ⟨x,hx,he⟩:=Exp2Decode.finite word hw hf
  rw [Model.Algebraic.toReal_of_toRat?_eq_some hx,he]

theorem positive_bound (word:Nat) (hw:word<0x7f800000) :
    ∃ out:Nat,Mufu.tanh word=some out ∧
      (Model.ofNatBits out:Model FloatFormat.binary32).isFinite=true ∧
      |(Model.ofNatBits out:Model FloatFormat.binary32).toReal-
        Real.tanh ((Model.ofNatBits word:Model FloatFormat.binary32).toReal)| ≤ 1/2^16 := by
  by_cases hs:word<119*2^23
  · have he:Mufu.exponent word ≤ 118 := by unfold Mufu.exponent;omega
    have hb:=TanhLinear.small_bits word (by omega) he
    obtain ⟨x,hx,_⟩:=Exp2Decode.positive word hw
    refine ⟨word,hb,Model.Algebraic.isFinite_of_toRat?_eq_some hx,?_⟩
    have hav:=actual_value word (by omega) (by unfold Mufu.magnitude;omega)
    rw [hav]
    have h:=TanhLinear.small_uniform_error word (by omega) he
    rw [hb,Option.getD_some] at h
    exact h.trans (by norm_num)
  · by_cases hm:word<130*2^23
    · let e:Fin 11:=⟨word/2^23-119,by omega⟩
      let f:Fin (2^23):=⟨word%2^23,Nat.mod_lt _ (by decide)⟩
      have he:word=inputWord e f := by unfold inputWord;dsimp [e,f];omega
      rw [he]
      exact decoded_error e f
    · have ht:=TanhPlateau.positive_native_bound word (by omega) hw
      refine ⟨0x3f800000,ht.1,by decide,ht.2.le.trans (by norm_num)⟩

theorem native_sign (word:Nat) (hw:word<0x7f800000) :
    Mufu.tanh (2^31+word)=(Mufu.tanh word).map (fun x=>2^31+x) := by
  have hs:word<2^31 := by omega
  have hm:Mufu.magnitude word<0x7f800000 := by unfold Mufu.magnitude;omega
  unfold Mufu.tanh
  rw [RcpSigned.magnitude_sign,RcpSigned.exponent_sign,RcpSigned.fraction_sign,
    RcpSigned.sign_zero word hs,RcpSigned.sign_set word hs]
  simp only [not_lt.mpr hm.le,ite_false,Nat.zero_add]
  split_ifs <;> rfl

theorem all_finite (word:Nat) (hw:word<2^32) (hf:Mufu.magnitude word<0x7f800000) :
    ∃ out:Nat,Mufu.tanh word=some out ∧
      (Model.ofNatBits out:Model FloatFormat.binary32).isFinite=true ∧
      |(Model.ofNatBits out:Model FloatFormat.binary32).toReal-
        Real.tanh ((Model.ofNatBits word:Model FloatFormat.binary32).toReal)| ≤ 1/2^16 := by
  by_cases hs:word<2^31
  · exact positive_bound word (by unfold Mufu.magnitude at hf;omega)
  · let mag:=word%2^31
    have hm:mag<0x7f800000 := hf
    have he:word=2^31+mag := by dsimp [mag];omega
    obtain ⟨out,ho,hfinite,hbound⟩:=positive_bound mag hm
    obtain ⟨x,hx,_⟩:=Exp2Decode.positive mag hm
    refine ⟨2^31+out,?_,?_,?_⟩
    · rw [he,native_sign mag hm,ho];rfl
    · rw [Nat.add_comm,←RcpSigned.toggle_word,←Model.neg_eq_toggleSign_of_supportsSignedZero _ (by decide)]
      exact (Model.isFinite_neg _).trans hfinite
    · rw [he,TanhPlateau.sign_decode out hfinite,
        TanhPlateau.sign_decode mag (Model.Algebraic.isFinite_of_toRat?_eq_some hx),Real.tanh_neg]
      simpa only [neg_sub_neg,abs_sub_comm] using hbound

#print axioms input_decode
#print axioms input_fields
#print axioms core_value
#print axioms native_bits
#print axioms decoded_error
#print axioms actual_value
#print axioms positive_bound
#print axioms native_sign
#print axioms all_finite
end GB300Math.TanhCentralUniform
