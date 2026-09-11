import json, pathlib

C = []
def _src(s):
    # nbformat: every line keeps its own trailing newline, because Jupyter
    # concatenates the list with no separator. Splitting on "\n" alone welds
    # the whole cell into one line.
    lines = s.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]

def md(s):   C.append({"cell_type": "markdown", "metadata": {}, "source": _src(s)})
def code(s): C.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                       "outputs": [], "source": _src(s)})

md(r"""
# Ethiopic (Ge'ez) script coverage in modern tokenizers

Measures how tokenizers handle **Amharic** relative to English and Korean, using the
Universal Declaration of Human Rights as a parallel corpus.

Motivated by [huggingface/tokenizers#1975](https://github.com/huggingface/tokenizers/issues/1975),
an RFC proposing jamo decomposition as a pre-tokenizer for Korean, framed as
"a blueprint for compositional scripts." Ethiopic is a compositional script that the
RFC does not cover, and it turns out to be a considerably more extreme case than Korean.

**Why this notebook exists rather than a one-off script:** the naive version of this
benchmark reports a tokenizer that *silently destroys* Amharic as the most efficient one.
Section 3 is an integrity gate that must pass before any efficiency number is trusted.

## What is measured

| § | Measurement |
|---|---|
| 3 | Round-trip integrity (gate: results from failing tokenizers are not reported as efficiency) |
| 4 | Document-level token inflation vs English |
| 5 | Per-character cost by script |
| 6 | Vocabulary coverage: how many Ethiopic syllables are a single token |
| 7 | Structural awareness: do the 7 vowel orders of one consonant share representation |
| 8 | Practical cost: context window and API spend |

## Data provenance

UDHR translations from the [UDHR in Unicode](https://github.com/eric-muller/udhr) project
(`data/udhr/udhr_{amh,eng,kor}.xml`), extracted to plain text in `data/`.
Same document in every language, so token counts are directly comparable.
""")

code(r"""
import os, sys, json, unicodedata
from collections import defaultdict
from pathlib import Path

DATA = Path("data")
# Ship a local tiktoken cache so the notebook runs offline.
CACHE = Path("tkcache")
if CACHE.is_dir():
    os.environ["TIKTOKEN_CACHE_DIR"] = str(CACHE.resolve())

import tiktoken
from tokenizers import Tokenizer as HFTokenizer

ETHIOPIC = range(0x1200, 0x1380)   # Ethiopic block + supplement start
HANGUL   = range(0xAC00, 0xD7A4)   # Hangul syllables

print("tiktoken  ", tiktoken.__version__)
import tokenizers as _tk; print("tokenizers", _tk.__version__)
""")

md(r"""
## 1. Corpus

One caveat has to be handled before any measurement.

The UDHR Amharic text uses **U+1361 ETHIOPIC WORDSPACE (`፡`)** as its word separator.
That is traditional orthography; modern Amharic uses ASCII spaces. Measuring only the
original would overstate token counts, because the separator itself is an unseen
3-byte character repeated on every word boundary.

Both variants are carried through the notebook. **`amh` (space-normalized) is the
number reported**; `amh_raw` is kept so the difference is visible rather than hidden.
""")

code(r"""
def load(code_):
    return (DATA / f"udhr_{code_}.txt").read_text(encoding="utf-8")

CORPUS = {
    "eng":     load("eng"),
    "kor":     load("kor"),
    "amh_raw": load("amh"),
}
CORPUS["amh"] = CORPUS["amh_raw"].replace("፡", " ")   # ETHIOPIC WORDSPACE -> space

REPORT_LANGS = ["eng", "kor", "amh"]   # amh_raw shown separately

print(f"{'lang':9} {'chars':>7} {'utf8 bytes':>11} {'bytes/char':>11}")
for k, v in CORPUS.items():
    b = len(v.encode("utf-8"))
    print(f"{k:9} {len(v):7d} {b:11d} {b/len(v):11.2f}")

print("\nFirst line of each:")
for k in ("eng", "kor", "amh"):
    print(f"  {k}: {CORPUS[k].splitlines()[0][:78]}")
""")

md(r"""
## 2. Tokenizers under test

Two families:

- **tiktoken** — `cl100k_base` (GPT-4), `o200k_base` (GPT-4o). Byte-level BPE trained on
  predominantly English and code.
- **HuggingFace** — `xlm-roberta-base` (multilingual SentencePiece),
  `bert-base-multilingual-cased` (WordPiece), `gpt2` (byte-level BPE).

A uniform wrapper hides the API differences. Any tokenizer that fails to download is
skipped with a warning rather than aborting the run.
""")

code(r"""
class Tok:
    # Uniform interface over tiktoken and HuggingFace tokenizers.
    def __init__(self, name, label, kind):
        self.name, self.label, self.kind = name, label, kind
        if kind == "tiktoken":
            self._e = tiktoken.get_encoding(name)
            self.vocab_size = self._e.n_vocab
        else:
            self._e = HFTokenizer.from_pretrained(name)
            self.vocab_size = self._e.get_vocab_size()

    def encode(self, text):
        # add_special_tokens=False is essential. HF tokenizers prepend <s>/[CLS],
        # which makes every encoding start with the same id and never have
        # length 1 -- silently faking both "0% single-token coverage" and
        # "100% of rows share a first token".
        if self.kind == "tiktoken":
            return self._e.encode(text)
        return self._e.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self._e.decode(ids)

    def tokens(self, text):
        if self.kind == "tiktoken":
            return [self._e.decode([i]) for i in self.encode(text)]
        return self._e.encode(text, add_special_tokens=False).tokens

    def n(self, text):
        return len(self.encode(text))

SPECS = [
    ("cl100k_base",                          "GPT-4 (cl100k)",      "tiktoken"),
    ("o200k_base",                           "GPT-4o (o200k)",      "tiktoken"),
    ("xlm-roberta-base",                     "XLM-R",               "hf"),
    ("google-bert/bert-base-multilingual-cased", "mBERT",           "hf"),
    ("gpt2",                                 "GPT-2",               "hf"),
]

TOKS = []
for name, label, kind in SPECS:
    try:
        TOKS.append(Tok(name, label, kind))
        print(f"loaded  {label:22} vocab={TOKS[-1].vocab_size}")
    except Exception as e:
        print(f"SKIPPED {label:22} {type(e).__name__}: {str(e)[:70]}")
""")

md(r"""
## 3. Integrity gate — round-trip

**This section exists because of a real trap.**

`bert-base-multilingual-cased` reports Amharic at roughly *half* the token count of
English, which reads as excellent compression. It is not. Every Amharic word maps to
`[UNK]` and the text decodes back to the empty string. The tokenizer is not compressing
Amharic, it is deleting it.

A token-count benchmark without this check ranks the most broken tokenizer first.

Any tokenizer that fails round-trip is marked **LOSSY** and excluded from the
efficiency rankings below.
""")

code(r"""
SAMPLE = "የሰው ልጅ ሁሉ ሲወለድ ነጻና በክብርና"

def roundtrip(tok, text):
    # Returns (ok, decoded). Whitespace-insensitive: tokenizers legitimately renormalize it.
    try:
        dec = tok.decode(tok.encode(text))
    except Exception as e:
        return False, f"<decode error: {e}>"
    return "".join(dec.split()) == "".join(text.split()), dec

INTEGRITY = {}
print(f"input: {SAMPLE}\n")
print(f"{'tokenizer':22} {'amharic':>9}  {'decoded'}")
for t in TOKS:
    ok, dec = roundtrip(t, SAMPLE)
    INTEGRITY[t.label] = ok
    shown = dec[:42] if dec.strip() else "<EMPTY — text destroyed>"
    print(f"{t.label:22} {'OK' if ok else 'LOSSY':>9}  {shown}")

print("\nFull-document round-trip:")
for t in TOKS:
    ok_all = all(roundtrip(t, CORPUS[l])[0] for l in REPORT_LANGS)
    INTEGRITY[t.label] = INTEGRITY[t.label] and ok_all
    print(f"  {t.label:22} {'OK' if ok_all else 'LOSSY'}")

TRUSTED = [t for t in TOKS if INTEGRITY[t.label]]
print(f"\nTrusted for efficiency comparison: {[t.label for t in TRUSTED]}")
""")

code(r"""
# Show exactly how mBERT fails, if it loaded. This is the cautionary example.
mb = next((t for t in TOKS if t.label == "mBERT"), None)
if mb and not INTEGRITY["mBERT"]:
    print("mBERT token stream for the Amharic sample:")
    print(" ", mb.tokens(SAMPLE))
    print("\n  -> every word is [UNK]; the Amharic content never reaches the model.")
""")

md(r"""
## 4. Document-level token inflation

Same document, three languages. English is the baseline.
""")

code(r"""
rows = []
for t in TOKS:
    n_eng = t.n(CORPUS["eng"])
    row = {"tokenizer": t.label, "vocab": t.vocab_size, "ok": INTEGRITY[t.label]}
    for l in REPORT_LANGS + ["amh_raw"]:
        row[l] = t.n(CORPUS[l])
        row[l + "_x"] = row[l] / n_eng
    rows.append(row)

hdr = f"{'tokenizer':22} {'vocab':>7} {'eng':>6} {'kor':>6} {'kor x':>7} {'amh':>6} {'amh x':>7} {'amh/kor':>8} {'':>6}"
print(hdr); print("-" * len(hdr))
for r in rows:
    flag = "" if r["ok"] else "LOSSY"
    print(f"{r['tokenizer']:22} {r['vocab']:7d} {r['eng']:6d} {r['kor']:6d} "
          f"{r['kor_x']:6.2f}x {r['amh']:6d} {r['amh_x']:6.2f}x "
          f"{r['amh']/r['kor']:7.2f}x {flag:>6}")

print("\nEffect of the ETHIOPIC WORDSPACE normalization (amh_raw -> amh):")
for r in rows:
    print(f"  {r['tokenizer']:22} raw={r['amh_raw']:6d}  normalized={r['amh']:6d}  "
          f"({(1-r['amh']/r['amh_raw'])*100:.1f}% reduction)")
""")

md(r"""
## 5. Per-character cost by script

Document-level counts blend script cost with language morphology. This isolates the
script: the seven vowel orders of a single Ethiopic consonant, against comparable runs
of Hangul and Latin.

For a UTF-8 byte-level BPE with no learned merges, Ethiopic costs **3.00 tokens per
character** — one token per byte, since Ethiopic codepoints are 3 bytes in UTF-8.
That value is the signature of zero coverage.
""")

code(r"""
SAMPLES = {
    "Ethiopic ha-series (ሀሁሂሃሄህሆ)": "ሀሁሂሃሄህሆ",
    "Ethiopic la-series (ለሉሊላሌልሎ)": "ለሉሊላሌልሎ",
    "Hangul syllables":  "한글가나다라마",
    "Latin":             "abcdefg",
}
for t in TOKS:
    print(f"--- {t.label}")
    for label, s in SAMPLES.items():
        ids = t.encode(s)
        note = "  <- 1 token per UTF-8 byte (no coverage)" if len(ids)/len(s) == 3.0 else ""
        print(f"    {label:46} {len(s)} chars -> {len(ids):3d} tok  ({len(ids)/len(s):.2f}/char){note}")
""")

md(r"""
## 6. Vocabulary coverage

Two independent methods, because each alone can mislead.

**a) Empirical (works for every tokenizer).** Encode each Ethiopic syllable on its own
and count how many become exactly one token **that decodes back to that character**.

The decode half of that condition is load-bearing. A tokenizer mapping every unknown
character to `[UNK]` satisfies "one token" for the entire script and scores 100% while
representing none of it — which is exactly what mBERT does here. This is the third
distinct place in this notebook where a broken tokenizer outscores a working one, after
§3 (token counts) and §7 (shared prefixes). **Coverage means one token that round-trips,
never merely one token.**

**b) Byte-level vocabulary scan (tiktoken only).** Inspect raw token bytes for Ethiopic
UTF-8 sequences (`E1 88 80`–`E1 8D BF`). Catches tokens holding a *partial* sequence,
which a string-based scan would miss.

**c) Direct vocabulary membership (HF only).** SentencePiece prefixes standalone input
with the word-boundary marker `▁`, so a syllable can be present in the vocabulary and
still never encode to a single token on its own. Method (a) alone would score such a
tokenizer at 0% and call it uncovered. This checks the vocabulary directly, both bare
and marked.

⚠️ Method (c) is only meaningful for SentencePiece and WordPiece. **Byte-level BPE
(GPT-2) stores tokens in a remapped alphabet**, so Ethiopic bytes never appear as
Ethiopic characters and this method reports 0 for the wrong reason. For those, trust
(a) and (b).

Two further traps this section avoids:

- `add_special_tokens=False` is set in the wrapper. Without it every HF encoding begins
  with `<s>`/`[CLS]`, so no encoding is ever length 1 and coverage reads as 0% for every
  HF tokenizer regardless of the truth.
- Counting only *bare* vocabulary entries understates SentencePiece for the same
  word-boundary reason.
""")

code(r"""
# Assigned Ethiopic syllables in the main block.
SYLLABLES = []
for cp in range(0x1200, 0x1380):
    try:
        unicodedata.name(chr(cp)); SYLLABLES.append(chr(cp))
    except ValueError:
        pass
HANGUL_SAMPLE = [chr(c) for c in range(0xAC00, 0xAC00 + 400)]
print(f"assigned Ethiopic codepoints in U+1200-U+137F: {len(SYLLABLES)}")

def covered(t, ch):
    # "Covered" = encodes to ONE token that decodes back to the character.
    # The decode check is not optional: a tokenizer that maps every unknown
    # character to [UNK] satisfies len(ids)==1 for the whole script and would
    # otherwise score 100% while understanding none of it.
    ids = t.encode(ch)
    if len(ids) != 1:
        return False
    try:
        return t.decode(ids).strip() == ch
    except Exception:
        return False

print(f"\n{'tokenizer':22} {'eth 1-tok':>11} {'of':>5} {'%':>7}   {'hangul 1-tok':>13} {'%':>7}")
COVERAGE = {}
for t in TOKS:
    e1 = sum(1 for ch in SYLLABLES if covered(t, ch))
    h1 = sum(1 for ch in HANGUL_SAMPLE if covered(t, ch))
    COVERAGE[t.label] = e1
    print(f"{t.label:22} {e1:11d} {len(SYLLABLES):5d} {e1/len(SYLLABLES)*100:6.1f}%   "
          f"{h1:13d} {h1/len(HANGUL_SAMPLE)*100:6.1f}%")

# (c) Direct vocabulary membership, HF only.
# SentencePiece prefixes standalone input with the word-boundary marker U+2581,
# so a syllable can be in the vocabulary yet never encode to one token on its own.
# Checking the vocabulary directly separates "absent" from "not word-initial".
print("\nDirect vocabulary membership (HF tokenizers):")
for t in TOKS:
    if t.kind != "hf":
        continue
    v = t._e.get_vocab()
    bare   = sum(1 for ch in SYLLABLES if ch in v)
    marked = sum(1 for ch in SYLLABLES if ("▁" + ch) in v)
    anypos = sum(1 for ch in SYLLABLES if ch in v or ("▁" + ch) in v)
    contains = sum(1 for k in v if any(ord(c) in ETHIOPIC for c in k))
    print(f"  {t.label:22} bare={bare:4d}  with-marker={marked:4d}  either={anypos:4d}/{len(SYLLABLES)}"
          f"   vocab entries containing Ethiopic={contains:5d}")
""")

code(r"""
# (b) exact byte-level scan, tiktoken only
def has_ethiopic_bytes(b):
    return any(pair in b for pair in
               (b"\xe1\x88", b"\xe1\x89", b"\xe1\x8a", b"\xe1\x8b", b"\xe1\x8c", b"\xe1\x8d"))

for t in TOKS:
    if t.kind != "tiktoken":
        continue
    vals = t._e.token_byte_values()
    full = sum(1 for b in vals
               if any(ord(c) in ETHIOPIC for c in b.decode("utf-8", errors="ignore")))
    partial = sum(1 for b in vals if has_ethiopic_bytes(b))
    print(f"{t.label:22} vocab={t.vocab_size:6d}  "
          f"tokens containing a COMPLETE Ethiopic char={full:5d}  "
          f"tokens touching Ethiopic bytes={partial:5d}")
""")

md(r"""
## 7. Structural awareness

The RFC's real argument is not cost, it is **structure**: ሀ ሁ ሂ ሃ ሄ ህ ሆ are one
consonant across seven vowel orders, and a tokenizer that assigns them unrelated ids
cannot generalize across them.

This checks whether the vowel orders of each consonant share any leading token.

A caution on interpreting a shared prefix in byte-level BPE: `o200k` gives ሀ and ሁ the
same first token, but that token is the UTF-8 prefix `E1 88`, which spans
U+1200–U+123F — several unrelated consonants. That is **byte** structure, not **script**
structure, and the section below separates the two.
""")

code(r"""
# Row-based grouping: in U+1200-U+1357 each block of 8 is one consonant
# (7 vowel orders + a labialized form). Approximate for some rows; adequate here.
rows_by_consonant = defaultdict(list)
for ch in SYLLABLES:
    if 0x1200 <= ord(ch) < 0x1358:
        rows_by_consonant[(ord(ch) - 0x1200) // 8].append(ch)
FULL_ROWS = [v for v in rows_by_consonant.values() if len(v) >= 7]
print(f"consonant rows with >=7 vowel orders: {len(FULL_ROWS)}")

for t in TOKS:
    shared = single = 0
    for row in FULL_ROWS:
        first = [t.encode(c)[0] for c in row[:7]]
        if len(set(first)) == 1:
            shared += 1
        if all(covered(t, c) for c in row[:7]):   # covered(), not len()==1: see §6
            single += 1
    flag = "" if INTEGRITY[t.label] else "  [LOSSY: shared token is [UNK]]"
    print(f"{t.label:22} rows sharing a first token: {shared:3d}/{len(FULL_ROWS)}   "
          f"rows where every form is 1 token: {single:3d}/{len(FULL_ROWS)}{flag}")
""")

code(r"""
# Does a shared first token mean the tokenizer learned the CONSONANT, or just a byte prefix?
probe = next((t for t in TOKS if t.label == "GPT-4o (o200k)"), None)
if probe:
    print("o200k first-token id per syllable (ha-series vs a-series):")
    for ch in "ሀሁሃአኢኦ":
        ids = probe.encode(ch)
        print(f"  {ch} U+{ord(ch):04X} -> {ids}   utf8={ch.encode('utf-8').hex(' ')}")
    print("\n  ሀ/ሁ/ሃ share a first token, and so do አ/ኢ/ኦ, but the shared token is the")
    print("  2-byte UTF-8 prefix. U+1200-U+123F spans several consonants, so the grouping")
    print("  does not correspond to a linguistic unit.")
""")

md(r"""
## 8. Practical cost

Token inflation is not academic: it consumes context window and is billed directly.
""")

code(r"""
CTX = 128_000
print(f"How much Amharic fits in a {CTX:,}-token context, vs English?\n")
print(f"{'tokenizer':22} {'eng chars':>11} {'amh chars':>11} {'ratio':>8}")
for t in TRUSTED:
    e_per = t.n(CORPUS["eng"]) / len(CORPUS["eng"])
    a_per = t.n(CORPUS["amh"]) / len(CORPUS["amh"])
    print(f"{t.label:22} {int(CTX/e_per):11,} {int(CTX/a_per):11,} {e_per/a_per:7.2f}x")

print("\nRelative API spend for the same content (English = 1.00):")
for t in TRUSTED:
    print(f"  {t.label:22} amharic costs {t.n(CORPUS['amh'])/t.n(CORPUS['eng']):.2f}x")
""")

md(r"""
## 9. Summary
""")

code(r"""
print("=" * 78)
print("SUMMARY")
print("=" * 78)
for t in TOKS:
    status = "" if INTEGRITY[t.label] else "   [LOSSY - excluded from efficiency claims]"
    amh_x = t.n(CORPUS["amh"]) / t.n(CORPUS["eng"])
    print(f"\n{t.label} (vocab {t.vocab_size:,}){status}")
    print(f"  Ethiopic syllables that are a single token: {COVERAGE[t.label]}/{len(SYLLABLES)}")
    print(f"  Amharic vs English tokens: {amh_x:.2f}x")
    print(f"  Amharic vs Korean tokens:  {t.n(CORPUS['amh'])/t.n(CORPUS['kor']):.2f}x")
""")

md(r"""
## 10. Findings and limitations

### Findings

1. **GPT-4 and GPT-4o contain zero tokens for a complete Ethiopic character**, across
   100k and 200k vocabularies, for a script with over 100 million speakers. Amharic
   falls back to raw UTF-8 bytes.
2. **Amharic is a substantially worse case than Korean** — several times the inflation —
   in exactly the tokenizers the RFC targets.
3. **This is not inherent to the script.** XLM-R round-trips Amharic and reaches roughly
   parity with English, tokenizing common words whole. The problem is byte-level BPE
   trained on English-and-code-heavy corpora, not Ge'ez.
4. **Point 3 cuts against the RFC's central claim.** The RFC argues structural
   decomposition beats brute-force vocabulary expansion. For Ethiopic, vocabulary
   inclusion at about 1% of the vocabulary already achieves parity. Anyone citing this
   notebook in support of decomposition has to engage with that, not omit it.
5. **A shared first token is not evidence of structural awareness.** o200k groups
   syllables by UTF-8 byte prefix, which does not align with consonant identity.

### Limitations

- **One document.** UDHR is formal, legal register. Results may differ for
  conversational or technical text.
- **Translation effects.** Token ratios blend script encoding with translation length
  and morphology. §5 and §6 isolate the script; §4 does not.
- **Consonant grouping is approximate.** §7 uses 8-codepoint rows, which does not
  perfectly model labialized forms.
- **Tokenizer selection is partial.** Llama and Gemma are gated on the Hub and are not
  measured. Their coverage is unknown and would materially extend this.
- **Tokenization is not model quality.** Efficient tokenization does not establish that
  a model understands Amharic well. That needs downstream evaluation.

### Reproducing

```
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./fetch_tiktoken_cache.sh
./.venv/bin/jupyter nbconvert --to notebook --execute amharic_tokenization_measurement.ipynb
```

`data/` is committed. The tiktoken BPE ranks are fetched rather than vendored, to avoid
redistributing third-party data files. HF tokenizers (§2) download on first use.

This notebook is generated by `build_notebook.py`; edit the generator, not the notebook.
""")

nb = {
    "cells": C,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
out = pathlib.Path("/Users/birukgebrujember/projectArchive/research/tokenizers-amharic-integration/"
                   "measurement/amharic_tokenization_measurement.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out}  ({len(C)} cells, {out.stat().st_size} bytes)")
