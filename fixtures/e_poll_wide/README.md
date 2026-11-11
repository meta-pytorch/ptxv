# e_poll_wide

The cubins are local `ptxas -arch=sm_103a` compilations of `source.ptx`, with the
instruction replacements recorded in `patches.json`. The reference preserves the
state affected by this regression; the candidate removes that dependence.
The regression uses SASS instruction semantics without GPU replay.
