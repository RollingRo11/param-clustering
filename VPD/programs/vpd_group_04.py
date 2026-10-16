"""h.0.mlp.c_fc: on at the start of a web address (http, https, ftp, with or without a leading space)."""
from token_kinds67 import *

READS = ()


def step(carry, prefix, upstream):
    p = position(carry); s = prefix.token_strings[-1]
    return p, 1.0 if word(s) in ('http', 'https', 'ftp') else 0.0
