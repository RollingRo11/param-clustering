"""h.0.mlp.c_fc: on at a token that closes a bracket and opens a brace, like "){" or "[]{" (code and LaTeX)."""
from token_kinds67 import *

READS = ()


def step(carry, prefix, upstream):
    p = position(carry); s = prefix.token_strings[-1]
    t = s.strip()
    return p, 1.0 if t.endswith('{') and (t[:-1].endswith(')') or t[:-1].endswith(']')) else 0.0
