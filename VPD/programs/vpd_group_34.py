"""h.3.attn.o_proj: on at the first position, positions 1-3 while an opening content word continues, and end-of-text."""
from token_kinds67 import *

READS = ()


def step(carry, prefix, upstream):
    p = position(carry); s = prefix.token_strings[-1]
    return p, 1.0 if p == 0 or s == EOT or opening_run(prefix, 3) else 0.0
