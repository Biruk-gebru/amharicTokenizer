# tokenizers-amharic-integration

Research into how the [huggingface/tokenizers](https://github.com/huggingface/tokenizers)
ecosystem handles the Ethiopic (Ge'ez) script, and whether Amharic support is worth
contributing upstream.

## Background

RFC [huggingface/tokenizers#1975](https://github.com/huggingface/tokenizers/issues/1975)
proposes jamo decomposition as a pre-tokenizer for Korean, and describes itself as
"a blueprint for compositional scripts," naming Tibetan, Devanagari and Thai.

Ethiopic is a compositional script and is not on that list. It is an abugida: ሀ ሁ ሂ ሃ ሄ ህ ሆ
is one consonant across seven vowel orders. Unlike Hangul, Ethiopic has **no arithmetic
decomposition** in Unicode. Its syllables are precomposed atomic codepoints with no
canonical decomposition, so a lookup table is required rather than a formula.

Before writing any code for that, the question is whether the problem is real and whether
decomposition is actually the right fix. That is what `measurement/` answers.

## Layout

```
measurement/
  amharic_tokenization_measurement.ipynb   the benchmark (committed with outputs)
  build_notebook.py                        regenerates the notebook; source of truth
  fetch_tiktoken_cache.sh                  populates tkcache/ for offline runs
  data/                                    UDHR parallel corpus (amh / eng / kor)
  requirements.txt
```

## Running it

```
cd measurement
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./fetch_tiktoken_cache.sh
./.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    amharic_tokenization_measurement.ipynb
```

`data/` is committed. The tiktoken BPE ranks are **fetched rather than vendored**, to
avoid redistributing third-party data files; `fetch_tiktoken_cache.sh` pulls them into
`tkcache/`, which is gitignored. HuggingFace tokenizers download on first use.

> **Editing note:** `build_notebook.py` generates the `.ipynb`. Edit the generator, not
> the notebook — regenerating overwrites direct edits.

## Results

UDHR, same document in each language. `amh` is space-normalized (see notebook §1).

| Tokenizer | Vocab | Ethiopic syllables as 1 token | amh vs eng | amh vs kor |
|---|---|---|---|---|
| GPT-4 `cl100k_base` | 100,277 | **0 / 358** | 6.64x | 2.79x |
| GPT-4o `o200k_base` | 200,019 | **0 / 358** | 4.98x | 3.64x |
| GPT-2 | 50,257 | **0 / 358** | 6.56x | 1.31x |
| XLM-R | 250,002 | 84 / 358 (323 in vocab) | **0.94x** | 0.88x |
| mBERT | 119,547 | 0 / 358 | *(LOSSY, excluded)* | — |

## Findings so far

1. **GPT-4 (`cl100k_base`) and GPT-4o (`o200k_base`) contain zero tokens holding a
   complete Ethiopic character**, across 100k and 200k vocabularies. Amharic falls all
   the way back to raw UTF-8 bytes.
2. On the same UDHR document, Amharic costs roughly **5–6.6x** the tokens of English in
   those tokenizers, which is **about 3x worse than Korean**, the script the RFC is about.
3. **This is not inherent to Ge'ez.** XLM-R round-trips Amharic cleanly and reaches
   roughly parity with English, tokenizing common words whole, using about 1% of its
   vocabulary on the script.
4. Point 3 **cuts against the RFC's central claim** that decomposition beats vocabulary
   expansion. For Ethiopic, plain vocabulary inclusion already works. Any argument for
   decomposition has to engage with that rather than omit it.
5. A shared leading token is **not** evidence of structural awareness. `o200k` groups
   Ethiopic syllables by UTF-8 byte prefix, which spans several unrelated consonants.

## A methodological warning

`bert-base-multilingual-cased` appears to encode Amharic in *half* the tokens of English.
It does not. Every Amharic word becomes `[UNK]` and the text decodes to the empty string.
The tokenizer is deleting the content, not compressing it.

This was not a one-off. While building the notebook the same broken tokenizer came top
in **three** different metrics, each time for a different reason:

| Metric | How mBERT faked a win | Fix |
|---|---|---|
| Token count (§4) | destroyed text counts as compression | round-trip gate (§3) |
| Shared prefixes (§7) | every row shares `[CLS]` / `[UNK]` | `add_special_tokens=False`, integrity flag |
| Single-token coverage (§6) | `[UNK]` is one token, so 358/358 = 100% | coverage requires the token to **decode back** |

Two further measurement traps, both of which silently produced 0% coverage for
tokenizers that actually have good coverage:

- HF tokenizers prepend `<s>`/`[CLS]`, so `len(encode(ch)) == 1` is never true.
- SentencePiece prefixes standalone input with `▁`, so a syllable can be in the
  vocabulary yet never encode to one token alone (XLM-R: 323 present, 84 standalone).

**The transferable lesson:** in any tokenizer benchmark, a metric that does not verify
the content survived will rank the most broken tokenizer first. Round-trip is not a
nice-to-have, it is the precondition for every other number on the page.

## Status

Measurement complete; no upstream contribution made yet. Finding 4 is the open question:
whether decomposition is worth proposing for Ethiopic at all, given that vocabulary
inclusion demonstrably works.

## Data

UDHR translations from the [UDHR in Unicode](https://github.com/eric-muller/udhr) project.
