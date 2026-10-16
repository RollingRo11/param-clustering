"""h.1.attn.v_proj: same trigger as group 17 ("there is / there are")."""
from token_kinds67 import *

READS = ()


def step(carry, prefix, upstream):
    p = position(carry); s = prefix.token_strings[-1]
    if word(s) != 'there': return p, 0.0
    if s.strip() == 'There': return p, 1.0
    before = prefix.token_strings[-2] if p else ''
    return p, 1.0 if is_punct(before) or word(before) in CLAUSE_START else 0.0
