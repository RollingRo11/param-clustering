"""h.0.mlp.c_fc: on at the digit "0" when it starts a number in running text or a quiz option (after "-", "(c)",
"to", "?"), off when it is an index or subscript (after "[", "_", "_{", "=", "\\{")."""
from token_kinds67 import *

READS = ()


def step(carry, prefix, upstream):
    p = position(carry); s = prefix.token_strings[-1]
    if s.strip() != '0': return p, 0.0
    before = prefix.token_strings[-2].rstrip() if p else ''
    if before.endswith(('[', '_', '_{', '=', '\\{')): return p, 0.0
    return p, 1.0
