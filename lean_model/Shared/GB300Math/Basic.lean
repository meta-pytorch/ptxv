namespace GB300Math

abbrev Word32 := BitVec 32

inductive Operation where
  | ex2Approx
  | lg2Approx
  | tanhApprox
  deriving DecidableEq, Repr

inductive SubnormalMode where
  | defaultBehavior
  | ftz
  deriving DecidableEq, Repr

structure Instruction where
  operation : Operation
  subnormals : SubnormalMode
  deriving DecidableEq, Repr

abbrev UnaryModel := Word32 → Word32

structure Observation where
  input : Word32
  output : Word32
  deriving DecidableEq, Repr

def AgreesOn (model : UnaryModel) (observations : List Observation) : Prop :=
  ∀ observation ∈ observations, model observation.input = observation.output

end GB300Math
