"""Hand-written token kinds shared by the VPD group programs (vpd_group_NN.py). Nothing here is read from the model."""
EOT = '<|endoftext|>'

# Very common English function words (lower case, no leading space).
COMMON = {'the', 'to', 'of', 'a', 'and', 'with', 'on', 'from', 'was', 'have', 'in', 'is', 'for', 'by', 'at', 'as', 'an'}
FUNCTION = COMMON | {'it', 'but', 'if', 'one', 'two', 'not', 'that', 'which', 'while', 'without', 'he', 'she', 'they',
                     'we', 'this', 'these', 'those', 'be', 'are', 'were', 'or', 'so', 'when', 'where', 'there', 'their',
                     'its', 'his', 'her', 'also', 'can', 'will', 'would', 'should', 'may', 'all', 'some', 'any', 'no'}
# Words that start a new clause, after which " there" is usually "there is / there are".
CLAUSE_START = {'that', 'and', 'while', 'but', 'if', 'because', 'so', 'where', 'when', 'although', 'though', 'since',
                'as', 'whether', 'unless', 'until', 'then', 'now', 'yet', 'or'}


def word(s):
    return s.strip().lower()


def has_letter(s):
    return any(c.isalpha() for c in s)


def is_punct(s):
    """No letters and no digits: punctuation, brackets, whitespace, newlines."""
    return not any(c.isalnum() for c in s)


def is_digit(s):
    return any(c.isdigit() for c in s) and not has_letter(s)


def is_function(s):
    return word(s) in FUNCTION


def is_content(s):
    """A word or word piece that is not a function word."""
    return has_letter(s) and not is_function(s)


def position(carry):
    """Index of the current token in the context (the groups do not reset at <|endoftext|>)."""
    return 0 if carry is None else carry + 1


def opening_run(prefix, last):
    """True at positions 1..last while the context opened with a content word and every later token is a word piece."""
    toks = prefix.token_strings; p = len(toks) - 1
    return 1 <= p <= last and is_content(toks[0]) and all(has_letter(t) for t in toks[1:p + 1])
