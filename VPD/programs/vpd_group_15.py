"""h.0.mlp.down_proj: at the first position, on when the first token is punctuation, whitespace, a function word
or a single letter; off for content words and numbers."""
from token_kinds67 import *

READS = ()


def step(carry, prefix, upstream):
    p = position(carry); s = prefix.token_strings[-1]
    return p, 1.0 if p == 0 and (is_punct(s) or is_function(s) or len(s.strip()) <= 1 and has_letter(s)) else 0.0
