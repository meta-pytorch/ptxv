namespace GB300Math.SparseMetadata

inductive Order where
  | legacy
  | ordered
  deriving Repr, BEq

def validCode (code : Nat) (tf32 : Bool) (order : Order) : Bool :=
  code < 16 && if tf32 then code == 4 || code == 14 else
    match order with
    | .legacy => code % 4 != code / 4
    | .ordered => code % 4 < code / 4

def validWords (metadata : Array Nat) (tf32 : Bool) (order : Order) : Bool :=
  metadata.size == 32 && metadata.all fun word =>
    word < 2^32 && (Array.range 8).all fun j => validCode (word / 2^(4*j) % 16) tf32 order

theorem legacy_accepts_distinct_indices :
    ((Array.range 16).filter fun code => validCode code false .legacy) =
      #[1,2,3,4,6,7,8,9,11,12,13,14] := by decide +kernel

theorem ordered_requires_increasing_indices :
    ((Array.range 16).filter fun code => validCode code false .ordered) =
      #[4,8,9,12,13,14] := by decide +kernel

theorem duplicate_indices_rejected :
    (#[0,5,10,15].all fun code =>
      !validCode code false .legacy && !validCode code false .ordered) = true := by decide +kernel

theorem tf32_rule_is_unchanged :
    ((Array.range 16).filter fun code => validCode code true .legacy) = #[4,14] ∧
    ((Array.range 16).filter fun code => validCode code true .ordered) = #[4,14] := by decide +kernel

#print axioms legacy_accepts_distinct_indices
#print axioms ordered_requires_increasing_indices
#print axioms duplicate_indices_rejected
#print axioms tf32_rule_is_unchanged

end GB300Math.SparseMetadata
