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
  hangul.py                                Korean jamo split (the RFC's own case), control
  decomposition_experiment.py              trains BPE raw vs decomposed, 6 budgets
  robustness_experiment.py                 same, cleaned corpus, FLORES+ eval, 3 seeds, 3 models
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
geo on Meta's behalf. Worth knowing before reproducing this.

**Aya Expanse is CC-BY-NC-4.0, non-commercial only.** If this work ever becomes
commercial, that row has to come out.

> **Editing note:** `build_notebook.py` generates the `.ipynb`. Edit the generator, not
> the notebook, since regenerating overwrites direct edits.

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
| mBERT | 119,547 | 0 / 358 | *lossy on Amharic, excluded* | n/a |

On GPT-4 the same content costs **6.6x** (UDHR) to **7.8x** (FLORES+) more tokens in
Amharic, so a 128k window holds about a seventh as much Amharic as English.

Figures in [`visuals/`](visuals/).

> The coverage column understates SentencePiece models: it counts characters that encode
> to one token *standalone*, and SentencePiece prefixes standalone input with `▁`.
> Amharic RoBERTa has all 358 in vocabulary and the best token efficiency measured.
> Notebook §6 reports both. Two further results look like bugs and are not: Qwen3 gets
> *worse* when space-normalized, and Gemma 2 emits exactly one token per character.
> Both are explained in §1.

## Does decomposition help? Barely, and not in byte-level BPE.

RFC #1975 argues decomposition beats vocabulary expansion. `measurement/ethiopic.py`
implements it for Ethiopic (259 of 358 syllables, round-trip clean), and `hangul.py`
does the RFC's own Korean jamo split as a control.

**As a pre-tokenizer on existing tokenizers:** worse for 22 of 22, from 1.75x to 6.04x
more tokens. Expected, since their vocabularies were learned on composed text.

**Retrained from scratch** (`robustness_experiment.py`): cleaned and deduplicated
Wikipedia for training, FLORES+ dev for evaluation, three seeds, budgets 1k to 32k.
Decomposed tokens / raw tokens, so below 1.0 means decomposition wins:

| Model | Korean | Amharic (full) | Amharic (lean) |
|---|---|---|---|
| byte-level BPE | 1.00 to 1.05 | 1.03 to 1.14 | ~1.005 |
| Unigram | 0.98 to 1.01 | 1.01 to 1.03 | **0.96 to 0.97** |
| char-level BPE (4k+) | 0.99 to 1.01 | ~1.00 | ~1.00 |

"Lean" keeps the 6th-order form as one character and splits the other six orders.

- In byte-level BPE, which most LLM tokenizers use, decomposition never wins for either
  script. Both arms start from the same 256 bytes, so there are no alphabet slots to save.
- In character-level models it helps in narrower ways: raw Korean cannot be trained below
  ~2.3k vocab at all, Korean `<unk>` drops ~10x (0.27% to 0.02%), and lean Ethiopic is
  ~3% shorter under Unigram at every budget from 2k up.
- Every effect is a few percent. The gap to English in the table above is up to 8x, and
  it tracks whether the vocabulary covers the script, not how the script is split.

Cells where a raw character-level model ends up over budget (its alphabet is bigger than
the vocabulary) are flagged by the script and not counted. The earlier single-seed
byte-level run is still in `decomposition_experiment.py`.

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
Amharic RoBERTa, which round-trips Amharic perfectly but cannot represent Korean, was
marked LOSSY and dropped from every Amharic result. The best Amharic tokenizer in the
study was excluded on the basis of a language it never claimed to support. Integrity is
now tracked per language.

**The transferable lesson:** in any tokenizer benchmark, a metric that does not verify
the content survived will rank the most broken tokenizer first, and a gate applied too
broadly will discard the best specialist. Round-trip is the precondition for every other
number on the page, and it has to be per language.

### §10 enforces this

The notebook builds three deliberately broken tokenizers (`DropScript`, `UnkMapper`,
`Prefixer`), each modelled on a real bug above, and asserts every metric catches them.
It also demonstrates the traps: the naive `len(ids)==1` test scores `UnkMapper` at
358/358 while `covered()` scores it 0. **The cell raises on failure**, so the claims here
are only as good as those guards, and the notebook will not execute if one stops working.

## Status

Measurement complete. No code contribution planned: a table-driven split can already be
expressed with the existing `Replace` / `Precompiled` normalizers, and the gains are small.
Results posted on RFC #1975.

## Data

UDHR translations from the [UDHR in Unicode](https://github.com/eric-muller/udhr) project.
