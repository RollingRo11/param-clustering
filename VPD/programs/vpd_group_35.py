"""h.3.mlp.down_proj: on at the article "an" / "An" (and the rare "az"), any spacing."""
from token_kinds67 import *

READS = ()


def step(carry, prefix, upstream):
    p = position(carry); s = prefix.token_strings[-1]
    return p, 1.0 if word(s) in ('an', 'az') else 0.0
