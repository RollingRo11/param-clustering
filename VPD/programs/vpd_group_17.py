"""h.1.attn.k_proj: on at "there" used as "there is / there are": capitalised "There", or " there" right after
punctuation or a clause-opening word (that, and, while, ...)."""
from token_kinds67 import *

READS = ()


def step(carry, prefix, upstream):
    p = position(carry); s = prefix.token_strings[-1]
    if word(s) != 'there': return p, 0.0
    if s.strip() == 'There': return p, 1.0
    before = prefix.token_strings[-2] if p else ''
    return p, 1.0 if is_punct(before) or word(before) in CLAUSE_START else 0.0
