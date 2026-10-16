"""h.0.mlp.down_proj: same trigger as group 4, on at "http" / "https" / "ftp"."""
from token_kinds67 import *

READS = ()


def step(carry, prefix, upstream):
    p = position(carry); s = prefix.token_strings[-1]
    return p, 1.0 if word(s) in ('http', 'https', 'ftp') else 0.0
