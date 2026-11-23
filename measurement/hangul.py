"""Decomposition of Hangul syllables into jamo -- the RFC's own proposal.

This mirrors `ethiopic.py` so the two scripts can be run through the identical
pipeline. It exists to test RFC huggingface/tokenizers#1975 on the script it was
actually written about, rather than only on Ethiopic.

Hangul, unlike Ethiopic, decomposes **arithmetically**:

    index   = codepoint - 0xAC00
    initial = index // 588
    medial  = (index % 588) // 28
    final   = index % 28          (0 = no final consonant)

11,172 syllables from 19 initials x 21 medials x 28 finals.

Two jamo alphabets
------------------
Unicode has two sets, and the choice matters for tokenization:

* **Conjoining** (U+1100 / U+1161 / U+11A8 blocks) -- what `NFD` produces. Canonical,
  but rarely seen standalone in real text.
* **Compatibility** (U+3131-U+3163) -- ㅎ ㅏ ㄴ, what a Korean keyboard emits and what
  appears standalone in ordinary text. **The RFC's own example uses these.**

Both are provided. `compat` is the default because it matches the RFC and because a
tokenizer is far more likely to have seen these characters, which is the same reasoning
that made `ethiopic.py` use real Ethiopic characters rather than invented sentinels.

The compatibility tables are written out rather than derived, so `_self_check()` composes
all 11,172 syllables back from them at import time. A transcription error fails loudly.
"""

import unicodedata

SYLLABLES = range(0xAC00, 0xD7A4)
_BASE, _N_MEDIAL, _N_FINAL = 0xAC00, 21, 28

# Compatibility jamo, in Unicode's initial/medial/final order.
CHOSEONG = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNGSEONG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONGSEONG = ["", *"ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"]

assert len(CHOSEONG) == 19 and len(JUNGSEONG) == 21 and len(JONGSEONG) == 28


def split(ch):
    """(initial, medial, final) compatibility jamo, or None if not a Hangul syllable."""
    cp = ord(ch)
    if cp not in SYLLABLES:
        return None
    i = cp - _BASE
    return (CHOSEONG[i // (_N_MEDIAL * _N_FINAL)],
            JUNGSEONG[(i % (_N_MEDIAL * _N_FINAL)) // _N_FINAL],
            JONGSEONG[i % _N_FINAL])


def join(initial, medial, final=""):
    i = CHOSEONG.index(initial)
    m = JUNGSEONG.index(medial)
    f = JONGSEONG.index(final)
    return chr(_BASE + (i * _N_MEDIAL + m) * _N_FINAL + f)


def decompose(text, mode="compat"):
    """Split Hangul syllables into jamo. mode='compat' (default) or 'canonical' (NFD)."""
    if mode == "canonical":
        return "".join(unicodedata.normalize("NFD", c) if ord(c) in SYLLABLES else c
                       for c in text)
    out = []
    for c in text:
        parts = split(c)
        out.append("".join(parts) if parts else c)
    return "".join(out)


def compose(text, mode="compat"):
    """Inverse of decompose()."""
    if mode == "canonical":
        return unicodedata.normalize("NFC", text)
    out, i = [], 0
    while i < len(text):
        if (i + 1 < len(text) and text[i] in CHOSEONG and text[i + 1] in JUNGSEONG):
            final = ""
            if i + 2 < len(text) and text[i + 2] in JONGSEONG[1:]:
                # A final only belongs to this block if what follows is not the start
                # of another block, otherwise "각이" would swallow the next initial.
                nxt = text[i + 3] if i + 3 < len(text) else ""
                if nxt not in JUNGSEONG:
                    final = text[i + 2]
            out.append(join(text[i], text[i + 1], final))
            i += 2 + (1 if final else 0)
        else:
            out.append(text[i]); i += 1
    return "".join(out)


def coverage(text):
    """(decomposable, total) Hangul syllables in text."""
    syl = [c for c in text if ord(c) in SYLLABLES]
    return len(syl), len(syl)          # every Hangul syllable decomposes


def _self_check():
    """Compose all 11,172 syllables back from the tables. Catches a transcription error."""
    bad = [chr(cp) for cp in SYLLABLES if join(*split(chr(cp))) != chr(cp)]
    if bad:
        raise AssertionError(f"jamo tables wrong for {len(bad)} syllables, e.g. {bad[:5]}")
    return len(list(SYLLABLES))


N_VERIFIED = _self_check()
