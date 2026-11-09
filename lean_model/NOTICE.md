# Source attribution

The original FP32 math-unit reconstruction and interpolation coefficients come
from Bram Wasti's ptxv snapshot `6e51c271debb39a1d943c875bfdb21a6150bd990`.

`Shared/GB300Math/TextureBC6Tables.lean` contains BC6H layouts extracted from
iOrange's bcdec, which offers MIT or public-domain licensing. Its source note is
preserved.

`Shared/GB300Math/TextureBC7Tables.lean` contains tables from Richard Geldreich
Jr.'s `bc7decomp_ref.cpp` (2020), under its public-domain option. Its source note
is preserved.

FloatLib and Mathlib are external dependencies of optional proof modules and
are not included in this directory. No license for those projects is replaced
by this notice.
