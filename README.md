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
visuals/                                   generated figures (notebook §11)
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

19 tokenizers, UDHR parallel corpus (same document in Amharic, English, Korean).
`amh` is space-normalized (notebook §1). Coverage = characters that encode to one
token **that decodes back**.

| Tokenizer | Vocab | Ethiopic 1-token | amh vs eng | Amharic in a 128k context |
|---|---|---|---|---|
| GPT-NeoX | 50,277 | 0 / 358 | 6.91x | 49,384 chars |
| GPT-4 | 100,277 | **0 / 358** | 6.64x | 53,119 chars |
| OLMo 2 | 100,278 | 0 / 358 | 6.64x | 53,119 chars |
| SmolLM2 | 49,152 | 0 / 358 | 6.58x | 52,441 chars |
| GPT-2 | 50,257 | 0 / 358 | 6.56x | 52,441 chars |
| Mistral 7B | 32,768 | 0 / 358 | 5.51x | 57,764 chars |
| DeepSeek-V3 | 128,815 | 0 / 358 | 5.20x | 68,503 chars |
| GPT-4o | 200,019 | **0 / 358** | 4.98x | 70,752 chars |
| gpt-oss-20b | 200,019 | 0 / 358 | 4.98x | 70,752 chars |
| BLOOM | 250,680 | 1 / 358 | 4.34x | 80,301 chars |
| Kimi K3 | 163,584 | 13 / 358 | 3.63x | 97,191 chars |
| Qwen3.5 Amharic SFT | 248,077 | 20 / 358 | 3.59x | 95,449 chars |
| Qwen3 | 151,669 | 116 / 358 | 3.48x | 101,425 chars |
| Aya-101 | 250,100 | 39 / 358 | 1.32x | 206,466 chars |
| NLLB-200 | 256,204 | 176 / 358 | 0.96x | 326,080 chars |
| XLM-R | 250,002 | 84 / 358 | **0.94x** | 332,595 chars |
| AfroXLM-R | 250,002 | 84 / 358 | **0.94x** | 332,595 chars |
| Amharic RoBERTa | 32,000 | 41 / 358 (358 in vocab) | **0.21x** | 481,109 chars |
| mBERT | 119,547 | 0 / 358 | *lossy on Amharic, excluded* | — |

English gets ~696,000 characters into a 128k-token window. Amharic gets **53,000** on
GPT-4: about **8%** of the context, at **6.6x** the price.

Figures in [`visuals/`](visuals/): token inflation, coverage, per-character cost,
context window.

> The coverage column understates SentencePiece models. It counts characters that encode
> to one token *standalone*, and SentencePiece prefixes standalone input with `▁`.
> Amharic RoBERTa has all 358 in its vocabulary and encodes Amharic in the fewest tokens
> of any tokenizer measured. Notebook §6 reports both.

## Findings so far

1. **GPT-4 and GPT-4o contain zero tokens holding a complete Ethiopic character**, across
   100k and 200k vocabularies. Confirmed two independent ways, including a byte-level scan
   that would catch a token holding a partial UTF-8 sequence. Amharic falls back to raw bytes.
2. **The gap is not uniform, and "all frontier models have zero" would be wrong.**
   DeepSeek-V3, gpt-oss-20b, OLMo 2, SmolLM2, GPT-NeoX and GPT-2 are also at zero, but
   Qwen3 covers 116/358 and Kimi K3 covers 13/358. A vocabulary *string* scan reports 0
   for all of them, because byte-level BPE stores tokens in a remapped alphabet — that
   method is invalid for this family and the notebook flags the disagreement explicitly.
3. **A 250,680-entry multilingual vocabulary can still contain almost nothing.** BLOOM
   covers 1 of 358.
4. **Fine-tuning a model for Amharic does not fix its tokenizer.** `qwen3.5-4b-amharic-sft`
   carries 20/358 and still costs 3.59x English — it inherits the base tokenizer untouched.
   Weights were adapted; the vocabulary was not.
5. **This is not inherent to Ge'ez.** XLM-R and AfroXLM-R reach 0.94x English, NLLB-200
   0.96x, and a 32k Amharic-specific tokenizer reaches **0.21x** — five times *cheaper*
   than English. The script tokenizes well when it is in the training data.
6. **Point 5 cuts against the RFC's central claim** that decomposition beats vocabulary
   expansion. For Ethiopic, plain vocabulary inclusion already works, at roughly 1% of
   vocabulary. Any argument for decomposition has to engage with that rather than omit it.
7. A shared leading token is **not** evidence of structural awareness. `o200k` groups
   Ethiopic syllables by UTF-8 byte prefix, which spans several unrelated consonants.

### What it costs

English gets ~696,000 characters into a 128k-token context window. Amharic gets **53,000**
on GPT-4 — about **8%** — and is billed at **6.6x** for the same document.

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

A fifth trap ran the opposite way. The round-trip gate was applied **globally**, so
Amharic RoBERTa — which round-trips Amharic perfectly but cannot represent Korean — was
marked LOSSY and dropped from every Amharic result. The best Amharic tokenizer in the
study was excluded on the basis of a language it never claimed to support. Integrity is
now tracked per language.

**The transferable lesson:** in any tokenizer benchmark, a metric that does not verify
the content survived will rank the most broken tokenizer first — and a gate applied too
broadly will discard the best specialist. Round-trip is the precondition for every other
number on the page, and it has to be per language.

### §10 enforces this

The notebook builds three deliberately broken tokenizers (`DropScript`, `UnkMapper`,
`Prefixer`), each modelled on a real bug above, and asserts every metric catches them.
It also demonstrates the traps: the naive `len(ids)==1` test scores `UnkMapper` at
358/358 while `covered()` scores it 0. **The cell raises on failure**, so the claims here
are only as good as those guards, and the notebook will not execute if one stops working.

## Status

Measurement complete; no upstream contribution made yet. Finding 4 is the open question:
whether decomposition is worth proposing for Ethiopic at all, given that vocabulary
inclusion demonstrably works.

## Data

UDHR translations from the [UDHR in Unicode](https://github.com/eric-muller/udhr) project.
