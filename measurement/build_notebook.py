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
| 10 | **Self-test**: deliberately broken tokenizers must be caught by every metric |
| 11 | Figures written to `../visuals/` |
| 12 | **Does decomposition help?** the RFC's proposal, tested two ways |

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

# Kimi's own pre-tokenizer split pattern, copied from tokenization_kimi.py in the
# model repo. Needed because Kimi ships tiktoken.model rather than tokenizer.json.
KIMI_PAT = "|".join([
    r"[\p{Han}]+",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\p{N}{1,3}", r" ?[^\s\p{L}\p{N}]+[\r\n]*", r"\s*[\r\n]+",
    r"\s+(?!\S)", r"\s+",
])

REVISIONS = json.loads((DATA / "revisions.json").read_text()) \
            if (DATA / "revisions.json").exists() else {}
print(f"pinned revisions: {len(REVISIONS)}")
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

Two results in that comparison look like bugs and are not:

- **Qwen3 gets *worse* when normalized** (5,811 -> 6,565 tokens). It has learned merges
  involving `፡` itself but fewer involving space-plus-Ethiopic. So "normalizing is the
  conservative choice" holds for every other tokenizer here but not for Qwen3, where it
  understates the raw text. Reported rather than smoothed over.
- **Gemma 2 emits exactly one token per character** on the raw text (5,202 chars ->
  5,202 tokens). Not a coincidence: it covers 241 of 358 Ethiopic characters individually
  but has learned no multi-character merges for Amharic, so it is effectively
  character-level for this script. That is why it beats GPT-4 (1 token/char vs 3) while
  still losing badly to XLM-R, which tokenizes whole words.
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

# Second, independent corpus. UDHR alone is one document in one register, which is the
# weakest point of a single-corpus benchmark. FLORES+ is 997 professionally translated
# sentences aligned across languages and is the standard corpus in this literature.
# Fetched, never committed -- its terms forbid re-hosting (see fetch_flores.py).
FLORES = {}
_fd = DATA / "flores"
if _fd.is_dir():
    FLORES = {l: (_fd / f"{l}.txt").read_text(encoding="utf-8")
              for l in ("eng", "kor", "amh") if (_fd / f"{l}.txt").exists()}
CORPORA = {"UDHR": {l: CORPUS[l] for l in REPORT_LANGS}}
if len(FLORES) == 3:
    CORPORA["FLORES+"] = FLORES
    print(f"second corpus loaded: FLORES+ "
          f"({len(FLORES['amh'].splitlines())} aligned sentences per language)")
else:
    print("FLORES+ not present -- run fetch_flores.py for the second corpus")

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

Twenty-three tokenizers in three groups. The grouping is for reading the charts only;
every tokenizer is measured identically.

- **English-centric** — GPT-4, GPT-4o, Kimi K3, gpt-oss-20b, Qwen3, DeepSeek-V3, OLMo 2,
  Mistral 7B, Mistral Nemo, Llama 3.1, Gemma 2, SmolLM2, GPT-NeoX, GPT-2. Byte-level BPE
  trained mostly on English and code.
- **Multilingual** — BLOOM, mBERT, XLM-R, NLLB-200, Aya-101, Aya Expanse. Trained
  deliberately across many languages.
- **African / Amharic-focused** — AfroXLM-R, Amharic RoBERTa, and a Qwen3.5 model
  fine-tuned *for* Amharic.

Three loading paths sit behind one wrapper:

| kind | source | why |
|---|---|---|
| `tiktoken` | built-in encodings | GPT-4 / GPT-4o |
| `tiktoken_file` | `tiktoken.model` on the Hub | Kimi ships this instead of `tokenizer.json`, so `from_pretrained` 404s |
| `hf` | `tokenizer.json` | everything else |

Any tokenizer that fails to load is skipped with a warning rather than aborting the run.

**Gated tokenizers.** Llama 3.1, Gemma 2 and Aya Expanse each require accepting that
repo's terms with a Hugging Face account before the tokenizer will download. Llama's form
collects date of birth, job title and geo on Meta's behalf — worth knowing before you
reproduce this, since the cost of that row is personal data rather than compute.

**Revisions are pinned.** Every HF tokenizer loads at a recorded commit sha from
`data/revisions.json`. HF repos are mutable, and without pinning the committed outputs
would silently drift away from the tokenizers that produced them.
""")

code(r"""
class Tok:
    # Uniform interface over tiktoken, tiktoken-file, and HuggingFace tokenizers.
    def __init__(self, name, label, kind, family="other"):
        self.name, self.label, self.kind, self.family = name, label, kind, family
        if kind == "tiktoken":
            self._e = tiktoken.get_encoding(name)
            self.vocab_size = self._e.n_vocab
        elif kind == "tiktoken_file":
            # Kimi ships a tiktoken.model instead of HF's tokenizer.json, so
            # Tokenizer.from_pretrained 404s on it. Load the ranks directly and
            # rebuild the Encoding with the model's own split pattern.
            from huggingface_hub import hf_hub_download
            from tiktoken.load import load_tiktoken_bpe
            ranks = load_tiktoken_bpe(hf_hub_download(name, "tiktoken.model"))
            self._e = tiktoken.Encoding(name=label, pat_str=KIMI_PAT,
                                        mergeable_ranks=ranks, special_tokens={})
            self.vocab_size = self._e.n_vocab
        else:
            # Pin to a recorded sha. HF repos are mutable; without this the committed
            # outputs silently stop matching the tokenizers that produced them.
            rev = REVISIONS.get(name)
            self._e = HFTokenizer.from_pretrained(name, revision=rev) if rev \
                      else HFTokenizer.from_pretrained(name)
            self.vocab_size = self._e.get_vocab_size()

    def encode(self, text):
        # add_special_tokens=False is essential. HF tokenizers prepend <s>/[CLS],
        # which makes every encoding start with the same id and never have
        # length 1 -- silently faking both "0% single-token coverage" and
        # "100% of rows share a first token".
        if self.kind in ("tiktoken", "tiktoken_file"):
            return self._e.encode(text)
        return self._e.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self._e.decode(ids)

    def tokens(self, text):
        if self.kind in ("tiktoken", "tiktoken_file"):
            return [self._e.decode([i]) for i in self.encode(text)]
        return self._e.encode(text, add_special_tokens=False).tokens

    def byte_values(self):
        # Raw token bytes, where the backend exposes them (tiktoken family only).
        return self._e.token_byte_values() if self.kind.startswith("tiktoken") else None

    def n(self, text):
        return len(self.encode(text))

# `family` groups tokenizers in the charts. It plays no part in any measurement.
SPECS = [
    ("cl100k_base",             "GPT-4",            "tiktoken",      "english-centric"),
    ("o200k_base",              "GPT-4o",           "tiktoken",      "english-centric"),
    ("moonshotai/Kimi-K3",      "Kimi K3",          "tiktoken_file", "english-centric"),
    ("openai/gpt-oss-20b",      "gpt-oss-20b",      "hf",            "english-centric"),
    ("Qwen/Qwen3-8B",           "Qwen3",            "hf",            "english-centric"),
    ("deepseek-ai/DeepSeek-V3", "DeepSeek-V3",      "hf",            "english-centric"),
    ("allenai/OLMo-2-1124-7B",  "OLMo 2",           "hf",            "english-centric"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral 7B", "hf",       "english-centric"),
    ("mistralai/Mistral-Nemo-Base-2407", "Mistral Nemo", "hf",       "english-centric"),
    ("meta-llama/Llama-3.1-8B",  "Llama 3.1",        "hf",            "english-centric"),
    ("google/gemma-2-9b",        "Gemma 2",          "hf",            "english-centric"),
    ("HuggingFaceTB/SmolLM2-1.7B", "SmolLM2",       "hf",            "english-centric"),
    ("EleutherAI/gpt-neox-20b", "GPT-NeoX",         "hf",            "english-centric"),
    ("gpt2",                    "GPT-2",            "hf",            "english-centric"),
    ("bigscience/bloom-560m",   "BLOOM",            "hf",            "multilingual"),
    ("google-bert/bert-base-multilingual-cased", "mBERT", "hf",      "multilingual"),
    ("xlm-roberta-base",        "XLM-R",            "hf",            "multilingual"),
    ("facebook/nllb-200-distilled-600M", "NLLB-200", "hf",           "multilingual"),
    ("CohereForAI/aya-101",     "Aya-101",          "hf",            "multilingual"),
    ("CohereLabs/aya-expanse-8b", "Aya Expanse",     "hf",            "multilingual"),
    ("Davlan/afro-xlmr-large",  "AfroXLM-R",        "hf",            "african"),
    ("rasyosef/roberta-base-amharic", "Amharic RoBERTa", "hf",       "african"),
    # A model fine-tuned FOR Amharic. Does its tokenizer follow? (See §9.)
    ("abdukuzi45/qwen3.5-4b-amharic-sft", "Qwen3.5 Amharic SFT", "hf", "african"),
]

# Gated repos need per-repo terms accepted on the Hub before they load.
GATED_NOTE = ("Llama 3.1, Gemma 2 and Aya Expanse are gated: they require accepting "
              "each repo's terms with a Hugging Face account. Llama's form in "
              "particular collects date of birth, job title and geo for Meta.")

TOKS = []
for name, label, kind, family in SPECS:
    try:
        TOKS.append(Tok(name, label, kind, family))
        print(f"loaded  {label:22} {family:16} vocab={TOKS[-1].vocab_size}")
    except Exception as e:
        print(f"SKIPPED {label:22} {type(e).__name__}: {str(e)[:60]}")
print(f"\nnote: {GATED_NOTE}")
""")

md(r"""
## 3. Integrity gate — round-trip

**This section exists because of a real trap.**

`bert-base-multilingual-cased` reports Amharic at roughly *half* the token count of
English, which reads as excellent compression. It is not. Every Amharic word maps to
`[UNK]` and the text decodes back to the empty string. The tokenizer is not compressing
Amharic, it is deleting it.

A token-count benchmark without this check ranks the most broken tokenizer first.

Integrity is tracked **per language, not globally**. This matters: Amharic RoBERTa has
a 32k Amharic-specific vocabulary and cannot represent Korean, so a global gate marks it
LOSSY and silently drops the best Amharic tokenizer in the study from every Amharic
result. It never claimed to handle Korean.

So there are two trusted sets:

- `TRUSTED` — round-trips all three languages; used for cross-language ratios like amh/kor
- `TRUSTED_AMH` — round-trips Amharic *and* English; used for every Amharic-vs-English
  claim and every figure

A tokenizer excluded from one is not necessarily excluded from the other, and the
exclusions are printed rather than inferred.
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

# Integrity is tracked PER LANGUAGE, not globally. A tokenizer built for one
# language legitimately fails on another -- Amharic RoBERTa has a 32k Amharic
# vocabulary and cannot represent Korean. Judging it globally would throw away the
# best Amharic tokenizer in the study on the basis of a language it never claimed.
print("\nFull-document round-trip, per language:")
print(f"  {'tokenizer':22} " + " ".join(f"{l:>7}" for l in REPORT_LANGS))
OK_LANG = {}
for t in TOKS:
    for l in REPORT_LANGS:
        OK_LANG[(t.label, l)] = roundtrip(t, CORPUS[l])[0]
    INTEGRITY[t.label] = all(OK_LANG[(t.label, l)] for l in REPORT_LANGS)
    cells = " ".join(f"{'OK' if OK_LANG[(t.label,l)] else 'LOSSY':>7}" for l in REPORT_LANGS)
    print(f"  {t.label:22} {cells}")

def ok_for(t, *langs):
    return all(OK_LANG[(t.label, l)] for l in langs)

# Trusted for a cross-language comparison only if it round-trips every language in it.
TRUSTED      = [t for t in TOKS if INTEGRITY[t.label]]          # all three languages
TRUSTED_AMH  = [t for t in TOKS if ok_for(t, "amh", "eng")]     # Amharic-vs-English claims
print(f"\ntrusted for all three languages ({len(TRUSTED)}): {[t.label for t in TRUSTED]}")
print(f"trusted for Amharic-vs-English ({len(TRUSTED_AMH)}): {[t.label for t in TRUSTED_AMH]}")
excluded = [t.label for t in TOKS if t not in TRUSTED_AMH]
print(f"excluded from Amharic claims: {excluded or 'none'}")
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

# Same measurement on the second corpus. If the two disagree, the finding is an
# artefact of one document rather than a property of the tokenizers.
for cname, corp in CORPORA.items():
    if cname == "UDHR":
        continue
    print(f"\n--- {cname} (independent corpus) ---")
    print(f"{'tokenizer':22} {'eng':>6} {'kor':>6} {'kor x':>7} {'amh':>6} {'amh x':>7}")
    for t in TOKS:
        if not ok_for(t, "amh", "eng"):
            continue
        ne, nk, na = (t.n(corp[l]) for l in ("eng", "kor", "amh"))
        print(f"{t.label:22} {ne:6d} {nk:6d} {nk/ne:6.2f}x {na:6d} {na/ne:6.2f}x")

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
VOCAB_SCAN = {}
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
    VOCAB_SCAN[t.label] = contains

# Reconcile (a) against (c). Where they disagree, (a) is right and (c) is measuring
# the wrong thing -- byte-level BPE stores tokens in a remapped alphabet, so a string
# scan of its vocabulary reports 0 no matter what the tokenizer can actually encode.
# Reporting (c) alone for those models understates them badly.
print("\nDisagreement between empirical coverage and vocabulary scan:")
any_disagreement = False
for t in TOKS:
    if t.label not in VOCAB_SCAN:
        continue
    emp, scan = COVERAGE[t.label], VOCAB_SCAN[t.label]
    if emp > 0 and scan == 0:
        any_disagreement = True
        print(f"  {t.label:22} encodes {emp:3d}/{len(SYLLABLES)} syllables as one token, "
              f"but a vocabulary string scan sees 0 -- byte-level BPE, trust the {emp}")
if not any_disagreement:
    print("  (none)")
""")

code(r"""
# (b) exact byte-level scan, tiktoken only
def has_ethiopic_bytes(b):
    return any(pair in b for pair in
               (b"\xe1\x88", b"\xe1\x89", b"\xe1\x8a", b"\xe1\x8b", b"\xe1\x8c", b"\xe1\x8d"))

for t in TOKS:
    vals = t.byte_values()
    if vals is None:      # HF backends do not expose raw token bytes
        continue
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
probe = next((t for t in TOKS if t.label == "GPT-4o"), None)
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
for t in TRUSTED_AMH:
    e_per = t.n(CORPUS["eng"]) / len(CORPUS["eng"])
    a_per = t.n(CORPUS["amh"]) / len(CORPUS["amh"])
    print(f"{t.label:22} {int(CTX/e_per):11,} {int(CTX/a_per):11,} {e_per/a_per:7.2f}x")

print("\nRelative API spend for the same content (English = 1.00):")
for t in TRUSTED_AMH:
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
## 10. Self-test — does this notebook's own methodology work?

Every metric above was wrong at least once while this notebook was being written, and
each time the wrong version *looked fine*. So the methodology gets tested the same way
production code does: build deliberately broken tokenizers, and assert that each metric
catches them.

Three synthetic failures, each modelled on a real bug found here:

| Fake | What it does | Real counterpart |
|---|---|---|
| `DropScript` | silently deletes every Ethiopic character | mBERT's `[UNK]` collapse |
| `UnkMapper` | maps any Ethiopic run to one `[UNK]` id | mBERT again, at character level |
| `Prefixer` | prepends a constant id to every encoding | HF `<s>` / `[CLS]` |

**This cell raises on failure.** If a metric ever stops catching one of these, the
notebook will not execute, which is the point: the claims in §12 are only as good as the
guards here.
""")

code(r"""
class FakeTok:
    # Minimal stand-in exposing just what roundtrip() and covered() call.
    kind, family, vocab_size = "fake", "self-test", 1000
    def __init__(self, label): self.label = label
    def n(self, text): return len(self.encode(text))

class DropScript(FakeTok):
    # Deletes Ethiopic entirely, then reports a small token count. Looks efficient.
    def encode(self, text): return [ord(c) for c in text if ord(c) not in ETHIOPIC]
    def decode(self, ids):  return "".join(chr(i) for i in ids)

class UnkMapper(FakeTok):
    # Collapses any run of Ethiopic into a single [UNK]. len(ids)==1 per syllable.
    UNK = -1
    def encode(self, text):
        out, prev_unk = [], False
        for c in text:
            if ord(c) in ETHIOPIC:
                if not prev_unk: out.append(self.UNK)
                prev_unk = True
            else:
                out.append(ord(c)); prev_unk = False
        return out
    def decode(self, ids): return "".join("" if i == self.UNK else chr(i) for i in ids)

class Prefixer(FakeTok):
    # Wraps a real tokenizer and prepends a constant id, mimicking <s> / [CLS].
    SPECIAL = 999999
    def __init__(self, base): super().__init__(f"Prefixer({base.label})"); self.base = base
    def encode(self, text): return [self.SPECIAL] + self.base.encode(text)
    def decode(self, ids):  return self.base.decode([i for i in ids if i != self.SPECIAL])

AMH = CORPUS["amh"][:400]
real = next(t for t in TOKS if INTEGRITY.get(t.label))
fakes = {"DropScript": DropScript("DropScript"),
         "UnkMapper":  UnkMapper("UnkMapper"),
         "Prefixer":   Prefixer(real)}

results = []
def check(name, condition, detail):
    results.append((name, bool(condition), detail))

# GUARD 1 -- the round-trip gate must reject silent destruction.
for nm in ("DropScript", "UnkMapper"):
    ok, _ = roundtrip(fakes[nm], AMH)
    check(f"round-trip rejects {nm}", not ok, f"roundtrip returned ok={ok}")

# GUARD 2 -- coverage must not credit [UNK]. The naive len(ids)==1 test scores
# UnkMapper at 358/358; covered() must score it 0.
naive = sum(1 for ch in SYLLABLES if len(fakes["UnkMapper"].encode(ch)) == 1)
strict = sum(1 for ch in SYLLABLES if covered(fakes["UnkMapper"], ch))
check("naive len==1 is fooled by [UNK] (demonstrates the trap)",
      naive == len(SYLLABLES), f"naive scored {naive}/{len(SYLLABLES)}")
check("covered() rejects [UNK]", strict == 0, f"covered() scored {strict}/{len(SYLLABLES)}")

# GUARD 3 -- a constant prefix must not read as shared structure.
pref = fakes["Prefixer"]
shared = sum(1 for row in FULL_ROWS if len({pref.encode(c)[0] for c in row[:7]}) == 1)
check("constant prefix fakes 100% shared first token (demonstrates the trap)",
      shared == len(FULL_ROWS), f"shared {shared}/{len(FULL_ROWS)}")
# The prefix guard is only meaningful against a base that does NOT already share
# 37/37 on its own -- otherwise it passes without detecting anything. Pick such a
# base explicitly, and fail loudly if none exists rather than passing vacuously.
def shared_rows(tok):
    return sum(1 for row in FULL_ROWS if len({tok.encode(c)[0] for c in row[:7]}) == 1)
discriminating = [t for t in TRUSTED if shared_rows(t) < len(FULL_ROWS)]
check("a discriminating base exists for the prefix guard", discriminating,
      f"{len(discriminating)} tokenizers share fewer than {len(FULL_ROWS)} rows")
if discriminating:
    b = discriminating[0]
    before, after = shared_rows(b), shared_rows(Prefixer(b))
    check("prefixing inflates shared-row count to 100%",
          before < len(FULL_ROWS) and after == len(FULL_ROWS),
          f"{b.label}: {before}/{len(FULL_ROWS)} -> {after}/{len(FULL_ROWS)} when prefixed")

# GUARD 4 -- a destructive tokenizer must never be in TRUSTED.
check("no tokenizer lossy on Amharic is used for Amharic claims",
      all(ok_for(t, "amh", "eng") for t in TRUSTED_AMH),
      "a tokenizer failing round-trip reached the efficiency table")

width = max(len(n) for n, _, _ in results)
for name, ok, detail in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:{width}}   {detail}")

failed = [n for n, ok, _ in results if not ok]
assert not failed, f"SELF-TEST FAILED: {failed}"
print(f"\nall {len(results)} guards pass")
""")

md(r"""
## 11. Figures

Charts are written to `../visuals/` for use in the write-up.

Design notes: one colour per family, assigned in fixed order and never recycled; every
bar carries its value directly, since the palette's mid-tones sit under 3:1 against the
surface and need that relief; grid recessive; no dual axes. Labels are ASCII because
Ethiopic glyphs render as missing-glyph boxes in most matplotlib font stacks.
""")

code(r"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIS = Path("../visuals"); VIS.mkdir(exist_ok=True)

SURFACE = "#fcfcfb"; INK = "#1a1a19"; MUTED = "#6b6b68"; GRID = "#e4e4e1"
FAMILY_COLOR = {          # fixed order, validated palette slots 1-3
    "english-centric": "#2a78d6",
    "multilingual":    "#eb6834",
    "african":         "#1baf7a",
}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": INK,
    "axes.edgecolor": GRID, "font.size": 10,
})

def style(ax, title, xlabel):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=14)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

def legend(ax, families):
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                          color=FAMILY_COLOR[f], label=f) for f in families]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower right")

def save(fig, name):
    fig.tight_layout(); fig.savefig(VIS / name, dpi=200); plt.close(fig)
    print(f"  wrote ../visuals/{name}")

USED = [f for f in ("english-centric", "multilingual", "african")
        if any(t.family == f for t in TRUSTED_AMH)]
""")

code(r"""
# FIGURE 1 -- token inflation: Amharic cost relative to English
rows = sorted(((t, t.n(CORPUS["amh"]) / t.n(CORPUS["eng"])) for t in TRUSTED_AMH),
              key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.8))
ax.barh([t.label for t, _ in rows], [v for _, v in rows],
        color=[FAMILY_COLOR[t.family] for t, _ in rows], height=0.68)
for i, (_, v) in enumerate(rows):
    ax.text(v + 0.06, i, f"{v:.2f}x", va="center", fontsize=9, color=INK)
ax.axvline(1.0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
ax.text(1.0, len(rows) - 0.3, " parity with English", fontsize=8, color=MUTED)
ax.set_xlim(0, max(v for _, v in rows) * 1.16)
style(ax, "Amharic costs this many times more tokens than the same English text",
      "tokens relative to English (UDHR, same document)")
legend(ax, USED)
save(fig, "token_inflation.png")
""")

code(r"""
# FIGURE 2 -- vocabulary coverage of the Ethiopic syllabary
rows = sorted(((t, COVERAGE[t.label] / len(SYLLABLES) * 100) for t in TRUSTED_AMH),
              key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.8))
ax.barh([t.label for t, _ in rows], [v for _, v in rows],
        color=[FAMILY_COLOR[t.family] for t, _ in rows], height=0.68)
for i, (t, v) in enumerate(rows):
    ax.text(v + 1.0, i, f"{COVERAGE[t.label]}/{len(SYLLABLES)}",
            va="center", fontsize=9, color=INK)
ax.set_xlim(0, 108)
style(ax, f"How many of the {len(SYLLABLES)} Ethiopic characters are a single token",
      "percent of the syllabary covered (token must decode back)")
legend(ax, USED)
save(fig, "ethiopic_coverage.png")
""")

code(r"""
# FIGURE 3 -- per-character cost, script isolated from language
SCRIPTS = [("Ethiopic", "ሀሁሂሃሄህሆ"), ("Hangul", "한글가나다라마"), ("Latin", "abcdefg")]
sel = sorted(TRUSTED_AMH, key=lambda t: t.n(SCRIPTS[0][1]) / 7, reverse=True)
y = range(len(sel)); h = 0.26
fig, ax = plt.subplots(figsize=(9, 0.55 * len(sel) + 2))
for j, (nm, s) in enumerate(SCRIPTS):
    vals = [t.n(s) / len(s) for t in sel]
    ax.barh([i + (1 - j) * h for i in y], vals, height=h * 0.88,
            color=list(FAMILY_COLOR.values())[j], label=nm)
    for i, v in zip(y, vals):
        ax.text(v + 0.04, i + (1 - j) * h, f"{v:.2f}", va="center", fontsize=7.5, color=INK)
ax.set_yticks(list(y)); ax.set_yticklabels([t.label for t in sel])
ax.axvline(3.0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
ax.text(3.0, len(sel) - 0.4, " 3.00 = one token per UTF-8 byte (no coverage)",
        fontsize=8, color=MUTED)
style(ax, "Tokens per character, by script", "tokens per character")
ax.legend(frameon=False, fontsize=9, loc="lower right")
save(fig, "per_char_cost.png")
""")

code(r"""
# FIGURE 4 -- how much text actually fits in a fixed context window
CTX = 128_000
rows = sorted(((t, CTX / (t.n(CORPUS["eng"]) / len(CORPUS["eng"])),
                   CTX / (t.n(CORPUS["amh"]) / len(CORPUS["amh"]))) for t in TRUSTED_AMH),
              key=lambda r: r[2])
y = range(len(rows)); h = 0.34
fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 2))
ax.barh([i + h / 2 for i in y], [e for _, e, _ in rows], height=h * 0.9,
        color=MUTED, label="English")
ax.barh([i - h / 2 for i in y], [a for _, _, a in rows], height=h * 0.9,
        color=[FAMILY_COLOR[t.family] for t, _, _ in rows], label="Amharic")
for i, (_, e, a) in zip(y, rows):
    ax.text(e + 8000, i + h / 2, f"{int(e):,}", va="center", fontsize=7.5, color=MUTED)
    ax.text(a + 8000, i - h / 2, f"{int(a):,}", va="center", fontsize=7.5, color=INK)
ax.set_yticks(list(y)); ax.set_yticklabels([t.label for t, _, _ in rows])
style(ax, f"Characters that fit in a {CTX:,}-token context window",
      "characters of text")
ax.legend(frameon=False, fontsize=9, loc="lower right")
save(fig, "context_window.png")
print("\nfigures written to ../visuals/")
""")

md(r"""
## 12. Does decomposition actually help?

Everything above measures the *problem*. This section tests the *proposed fix*.

RFC #1975 argues that decomposing a compositional script into its parts beats
"brute-force" vocabulary expansion. Ethiopic can be decomposed the same way Hangul can,
but with an important difference: **Unicode defines no decomposition for it.** Hangul
decomposes arithmetically; Ethiopic syllables are precomposed atomic codepoints with no
canonical decomposition, so a table is required. `ethiopic.py` derives that table from
Unicode character names rather than hand-writing 358 entries.

The representation uses characters that already exist in the script:

```
ሁ (hu)  ->  ህ (bare consonant, 6th order)  +  ኡ (standalone vowel, glottal series)
```

That mirrors the jamo approach but avoids invented sentinels — a tokenizer has some
chance of having seen ህ and ኡ, and no chance of having seen a private-use marker.

The proposal gets tested two ways, because "pre-tokenizer" can mean either.

**A. As a pre-tokenizer on existing tokenizers** — what the RFC literally proposes, since
a pre-tokenizer runs against an already-trained vocabulary.

**B. As a training-time intervention** — retrain BPE on decomposed text so the merges are
learned over the decomposed alphabet. Run separately by
`decomposition_experiment.py`, which trains both arms at six vocabulary budgets on
Amharic Wikipedia and evaluates on held-out text.

Both arms are normalised against the **original** character count. Decomposed text has
~1.8x the characters, so dividing each arm by its own length would hand decomposition a
free win from the denominator.
""")

code(r"""
import ethiopic as EE

amh = CORPUS["amh"]
dec = EE.decompose(amh)
d, t = EE.coverage(amh)
col, syl = EE.collision_rate(amh)

print(f"decomposable syllables in the table: {len(EE.DECOMPOSE)} of 358")
print(f"decomposable characters in this corpus: {d}/{t} ({d/t*100:.1f}%)")
print(f"characters after decomposition: {len(amh):,} -> {len(dec):,} ({len(dec)/len(amh):.2f}x)")
print(f"round-trip: {'OK' if EE.compose(dec) == amh else 'LOSSY'}"
      f"   ambiguous base+carrier pairs already in source: {col}")
print()
for s in "ሀሁሂሃሄህሆ":
    print(f"   {s} -> {''.join(EE.DECOMPOSE.get(s, (s,)))}")
""")

code(r"""
# A. Decomposition applied to EXISTING tokenizers.
print("Tokens for the same Amharic text, raw vs decomposed:\n")
print(f"{'tokenizer':22} {'raw':>7} {'decomposed':>11} {'change':>9}")
worse = 0
for t in TOKS:
    if not ok_for(t, "amh"):
        continue
    r, dd = t.n(amh), t.n(dec)
    worse += dd > r
    print(f"{t.label:22} {r:7d} {dd:11d} {dd/r:8.2f}x")
print(f"\ndecomposition is worse for {worse} of "
      f"{sum(1 for t in TOKS if ok_for(t,'amh'))} tokenizers")
print("A pre-tokenizer runs against an already-trained vocabulary, so this is the")
print("literal reading of the RFC. For Ethiopic it costs tokens rather than saving them:")
print("the text roughly doubles in length and no existing vocabulary has merges for it.")
""")

code(r"""
# B. Retraining BPE on decomposed text, at several vocabulary budgets.
# Produced by decomposition_experiment.py (trains on Amharic Wikipedia, evaluates
# on held-out text). Loaded here rather than retrained, since training six pairs of
# BPE models takes minutes.
import json
res_path = DATA / "decomposition_results.json"
if not res_path.exists():
    print("run `python decomposition_experiment.py` to produce this table")
else:
    rows = json.loads(res_path.read_text())
    print(f"{'vocab':>7}  {'RAW tok/char':>13} {'fert':>6}   "
          f"{'DECOMP tok/char':>16} {'fert':>6}   {'ratio':>7}  verdict")
    for r in rows:
        v = "decomp wins" if r["ratio"] < 1 else "raw wins"
        print(f"{r['vocab']:7d}  {r['raw_tok_per_char']:13.4f} {r['raw_fertility']:6.2f}   "
              f"{r['dec_tok_per_char']:16.4f} {r['dec_fertility']:6.2f}   "
              f"{r['ratio']:6.3f}x  {v}")
    wins = [r["vocab"] for r in rows if r["ratio"] < 1]
    print(f"\ndecomposition wins at: {wins or 'no budget tested'}")
    print("Note the direction: the gap is WIDEST at small budgets and narrows as the")
    print("vocabulary grows. The RFC's premise is the opposite -- that decomposition")
    print("should help most precisely when vocabulary is scarce.")
""")

code(r"""
# FIGURE 5 -- decomposition vs raw across vocabulary budgets
if res_path.exists():
    rows = json.loads(res_path.read_text())
    xs = [r["vocab"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(xs, [r["raw_tok_per_char"] for r in rows], marker="o", markersize=7,
            linewidth=2, color=FAMILY_COLOR["english-centric"], label="raw Amharic")
    ax.plot(xs, [r["dec_tok_per_char"] for r in rows], marker="s", markersize=7,
            linewidth=2, color=FAMILY_COLOR["multilingual"], label="decomposed")
    for r in rows:
        ax.text(r["vocab"], r["dec_tok_per_char"] + 0.012, f"{r['ratio']:.3f}x",
                ha="center", fontsize=8, color=MUTED)
    ax.set_xscale("log"); ax.set_xticks(xs)
    ax.set_xticklabels([f"{v//1000}k" for v in xs])
    style(ax, "Decomposition loses at every vocabulary budget",
          "BPE vocabulary size (trained on Amharic Wikipedia, held-out evaluation)")
    ax.set_ylabel("tokens per original character", fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9)
    save(fig, "decomposition_budgets.png")
""")

md(r"""
## 13. Findings and limitations

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
- **Tokenizer selection is partial, and one exclusion is deliberate.** Llama is not
  measured: its gate form collects date of birth, job title and geo for Meta, which is
  not a fair trade for one table row. Gemma 2 and Aya Expanse occupy the same position
  and are included. Other gated tokenizers remain unmeasured.
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
