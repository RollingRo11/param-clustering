"""h.3.attn.k_proj: on at the first position, at position 1 after an opening content word, and at end-of-text."""
from token_kinds67 import *

READS = ()


def step(carry, prefix, upstream):
    p = position(carry); s = prefix.token_strings[-1]
    return p, 1.0 if p == 0 or s == EOT or opening_run(prefix, 1) else 0.0
