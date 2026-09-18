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
  ethiopic.py                              Ethiopic decomposition table + round-trip
  decomposition_experiment.py              trains BPE raw vs decomposed, 6 budgets
  fetch_tiktoken_cache.sh                  tiktoken BPE ranks (gitignored)
  fetch_flores.py                          FLORES+ second corpus (gitignored, see below)
  fetch_corpus.py                          Amharic Wikipedia, for BPE training
  pin_revisions.py                         records exact HF commit shas
  data/                                    UDHR corpus, revisions.json, results
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

Full run, including the second corpus and the decomposition experiment:

```
./fetch_flores.py          # needs an HF token with FLORES+ terms accepted
./fetch_corpus.py          # Amharic Wikipedia, no token needed
python decomposition_experiment.py
```

**Nothing third-party is vendored.** The tiktoken BPE ranks, FLORES+, and the Wikipedia
corpus are all fetched and gitignored. FLORES+ especially: accepting its terms includes
*"I agree not to re-host FLORES+ in places where it could be picked up by web crawlers"*,
so it must never be committed here. It is also used **only** as held-out evaluation, never
for training, per its second term.

Gated tokenizers (Llama 3.1, Gemma 2, Aya Expanse) need each repo's terms accepted with
a Hugging Face account. Llama's form in particular collects date of birth, job title and
geo on Meta's behalf — worth knowing before reproducing this.

**Aya Expanse is CC-BY-NC-4.0, non-commercial only.** If this work ever becomes
commercial, that row has to come out.

> **Editing note:** `build_notebook.py` generates the `.ipynb`. Edit the generator, not
> the notebook — regenerating overwrites direct edits.

## Results

23 tokenizers, two independent parallel corpora: **UDHR** (one formal document) and
**FLORES+** (997 professionally translated sentences). Both corpora agree on the
ranking, which is the point of having two. Coverage = characters that encode to one
token **that decodes back**. Model revisions are pinned in `measurement/data/revisions.json`.

| Tokenizer | Vocab | Ethiopic 1-token | amh vs eng (UDHR) | amh vs eng (FLORES+) |
|---|---|---|---|---|
| GPT-NeoX | 50,277 | 0 / 358 | 6.91x | 7.96x |
| **GPT-4** | 100,277 | **0 / 358** | 6.64x | **7.76x** |
| **Llama 3.1** | 128,256 | **0 / 358** | 6.64x | **7.75x** |
| OLMo 2 | 100,278 | 0 / 358 | 6.64x | 7.76x |
| Mistral Nemo | 131,072 | 0 / 358 | 6.57x | 7.57x |
| SmolLM2 | 49,152 | 0 / 358 | 6.58x | 7.38x |
| GPT-2 | 50,257 | 0 / 358 | 6.56x | 7.56x |
| Mistral 7B | 32,768 | 0 / 358 | 5.51x | 6.28x |
| **Aya Expanse** | 255,029 | **0 / 358** | 5.36x | **6.19x** |
| DeepSeek-V3 | 128,815 | 0 / 358 | 5.20x | 6.11x |
| GPT-4o / gpt-oss-20b | 200,019 | 0 / 358 | 4.98x | 5.91x |
| BLOOM | 250,680 | 1 / 358 | 4.34x | 5.13x |
| Qwen3.5 Amharic SFT | 248,077 | 20 / 358 | 3.59x | 4.32x |
| Kimi K3 | 163,584 | 13 / 358 | 3.63x | 4.34x |
| Qwen3 | 151,669 | 116 / 358 | 3.48x | 4.15x |
| **Gemma 2** | 256,000 | **241 / 358** | **2.63x** | **3.10x** |
| Aya-101 | 250,100 | 39 / 358 | 1.32x | 1.75x |
| NLLB-200 | 256,204 | 176 / 358 | 0.96x | 1.34x |
| XLM-R / AfroXLM-R | 250,002 | 84 / 358 | **0.94x** | 1.35x |
| Amharic RoBERTa | 32,000 | 41 / 358 (358 in vocab) | **0.21x** | **0.32x** |
| mBERT | 119,547 | 0 / 358 | *lossy on Amharic, excluded* | — |

English gets ~696,000 characters into a 128k-token window. Amharic gets **53,000** on
GPT-4: about **8%** of the context, at **6.6x** the price.

Figures in [`visuals/`](visuals/).

> The coverage column understates SentencePiece models: it counts characters that encode
> to one token *standalone*, and SentencePiece prefixes standalone input with `▁`.
> Amharic RoBERTa has all 358 in vocabulary and the best token efficiency measured.
> Notebook §6 reports both. Two further results look like bugs and are not — Qwen3 gets
> *worse* when space-normalized, and Gemma 2 emits exactly one token per character.
> Both are explained in §1.

## Does decomposition help? No.

RFC #1975 argues decomposition beats vocabulary expansion. `measurement/ethiopic.py`
implements it for Ethiopic (259 of 358 syllables, 98.2% corpus coverage, round-trip
clean) and it was tested both ways the proposal can be read.

**As a pre-tokenizer on existing tokenizers** — the literal reading, since a
pre-tokenizer runs against an already-trained vocabulary:

> **worse for 22 of 22 tokenizers**, from 1.75x to 6.04x more tokens.

**Retrained from scratch on decomposed text**, six vocabulary budgets, trained on Amharic
Wikipedia, evaluated on held-out text, both arms normalized against the *original*
character count:

| Vocab | raw tok/char | decomp tok/char | result |
|---|---|---|---|
| 1,000 | 0.6029 | 0.6529 | raw wins 1.083x |
| 2,000 | 0.5074 | 0.5361 | raw wins 1.056x |
| 4,000 | 0.4399 | 0.4590 | raw wins 1.043x |
| 8,000 | 0.3936 | 0.4058 | raw wins 1.031x |
| 16,000 | 0.3627 | 0.3714 | raw wins 1.024x |
| 32,000 | 0.3345 | 0.3386 | raw wins 1.012x |

Note the **direction**: the gap is widest at small budgets and narrows as vocabulary
grows. The RFC's premise is the opposite — that decomposition should help most precisely
when vocabulary is scarce. For Ethiopic it converges toward parity from the wrong side
and never crosses.

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
