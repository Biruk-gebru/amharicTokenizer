"""Decomposition of Ethiopic (Ge'ez) syllables into consonant + vowel.

Ethiopic is an abugida: ሀ ሁ ሂ ሃ ሄ ህ ሆ is one consonant across seven vowel orders.
Unlike Hangul, Unicode defines **no decomposition** for it -- the syllables are
precomposed atomic codepoints with no canonical decomposition and no arithmetic
relationship between them. So a table is required.

The table is derived from Unicode character names rather than hand-written, because
the names already encode the structure ("ETHIOPIC SYLLABLE HU" = consonant H, vowel U).

Representation
--------------
A syllable maps to two characters that already exist in the script:

    base    the 6th-order (sadəs) form, which is the bare consonant     ሁ -> ህ
    carrier the matching vowel from the glottal series, a standalone vowel  u -> ኡ

    ሁ (hu)  ->  ህ + ኡ

This mirrors the jamo approach in huggingface/tokenizers#1975, but uses real Ethiopic
characters rather than invented sentinels. That matters: a tokenizer has some chance of
having seen ህ and ኡ, and no chance of having seen a private-use marker.

What is not decomposed
----------------------
Only series carrying all seven core orders are decomposed (37 of 52 series, 259 of 358
characters). Labialized forms (LWA, QWA, ...) have no clean consonant+vowel split and
are left atomic, as are the partial series (RYA, MYA, FYA) and all punctuation,
digits and numerals.

Invertibility
-------------
`compose(decompose(s)) == s` is NOT guaranteed in general: a base character followed in
the source text by a naturally-occurring carrier is indistinguishable from a decomposed
pair. `collision_rate()` measures how often that actually happens in a given corpus, so
the risk is quantified rather than assumed away.
"""

import unicodedata

ETHIOPIC_BLOCK = range(0x1200, 0x1380)
CORE_VOWELS = ["A", "U", "I", "AA", "EE", "E", "O"]   # orders 1-7; "E" is the bare form
_VOWEL_SUFFIXES = ["WAA", "WEE", "WOA", "OA", "WA", "WE", "WI", "WU",
                   "AA", "EE", "A", "U", "I", "E", "O"]


def _series():
    """{consonant_name: {vowel_name: char}} for every ETHIOPIC SYLLABLE codepoint."""
    out = {}
    for cp in ETHIOPIC_BLOCK:
        ch = chr(cp)
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if not name.startswith("ETHIOPIC SYLLABLE "):
            continue
        rest = name[len("ETHIOPIC SYLLABLE "):]
        if " " in rest:                      # e.g. "GLOTTAL A"
            cons, vowel = rest.rsplit(" ", 1)
        else:
            for v in _VOWEL_SUFFIXES:
                if rest.endswith(v) and len(rest) > len(v):
                    cons, vowel = rest[:-len(v)], v
                    break
            else:
                continue
        out.setdefault(cons, {})[vowel] = ch
    return out


SERIES = _series()
CARRIERS = {v: SERIES["GLOTTAL"][v] for v in CORE_VOWELS}   # A->አ U->ኡ I->ኢ ...

# syllable -> (base, carrier); only series with all seven core orders
DECOMPOSE = {}
for _cons, _forms in SERIES.items():
    if not all(v in _forms for v in CORE_VOWELS):
        continue
    _base = _forms["E"]                      # 6th order = bare consonant
    for _v in CORE_VOWELS:
        DECOMPOSE[_forms[_v]] = (_base, CARRIERS[_v])

COMPOSE = {pair: syl for syl, pair in DECOMPOSE.items()}
BASES = {b for b, _ in DECOMPOSE.values()}
CARRIER_SET = set(CARRIERS.values())


def decompose(text: str) -> str:
    """Split every decomposable Ethiopic syllable into base + vowel carrier."""
    return "".join("".join(DECOMPOSE[c]) if c in DECOMPOSE else c for c in text)


def compose(text: str) -> str:
    """Inverse of decompose(). Greedy: a base followed by a carrier recombines."""
    out, i = [], 0
    while i < len(text):
        pair = (text[i], text[i + 1]) if i + 1 < len(text) else None
        if pair in COMPOSE:
            out.append(COMPOSE[pair]); i += 2
        else:
            out.append(text[i]); i += 1
    return "".join(out)


def collision_rate(text: str):
    """How often compose() would wrongly merge a base+carrier already in the source.

    Returns (collisions, decomposable_syllables). A non-zero collision count means
    round-tripping this corpus is lossy and the scheme needs an escape mechanism.
    """
    collisions = sum(
        1 for i in range(len(text) - 1)
        if text[i] in BASES and text[i + 1] in CARRIER_SET
    )
    return collisions, sum(1 for c in text if c in DECOMPOSE)


def coverage(text: str):
    """(decomposable, total) Ethiopic characters in text."""
    eth = [c for c in text if ord(c) in ETHIOPIC_BLOCK]
    return sum(1 for c in eth if c in DECOMPOSE), len(eth)
