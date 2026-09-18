"""Does decomposition beat vocabulary expansion? Train BPE both ways and compare.

RFC huggingface/tokenizers#1975 argues that decomposing compositional scripts into
their parts beats "brute-force" vocabulary expansion. That claim is about how a
tokenizer is TRAINED, so testing it means training two tokenizers on the same corpus
-- one on raw text, one on decomposed text -- at several vocabulary budgets, and
comparing how many tokens each needs for held-out text.

Reported per budget:
  fertility  tokens per whitespace word (the standard metric in this literature)
  tok/char   tokens per character of the ORIGINAL text, so the two arms are
             comparable -- decomposed text has ~1.8x the characters, and dividing
             by its own length would flatter it for free.
"""

import json, random, sys
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

import ethiopic as E

BUDGETS = [1000, 2000, 4000, 8000, 16000, 32000]
SEED = 20260918


def load_corpus():
    p = Path("data/amh_wiki.txt")
    if not p.exists():
        sys.exit("data/amh_wiki.txt missing -- run fetch_corpus.py first")
    paras = [x.strip() for x in p.read_text(encoding="utf-8").split("\n\n") if x.strip()]
    random.Random(SEED).shuffle(paras)
    cut = int(len(paras) * 0.9)
    return paras[:cut], paras[cut:]          # train, held-out


def train_bpe(texts, vocab_size):
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def measure(tok, held_out, decompose):
    """Tokens needed for held-out text, normalised against the ORIGINAL characters."""
    n_tok = n_chars = n_words = 0
    for t in held_out:
        src = E.decompose(t) if decompose else t
        n_tok += len(tok.encode(src, add_special_tokens=False).ids)
        n_chars += len(t)                    # original length, both arms
        n_words += len(t.split())
    return n_tok, n_tok / n_chars, n_tok / max(n_words, 1)


def main():
    train, held = load_corpus()
    train_raw = train
    train_dec = [E.decompose(t) for t in train]
    tr_chars = sum(len(t) for t in train)
    print(f"corpus: {len(train)} train paragraphs ({tr_chars:,} chars), "
          f"{len(held)} held-out ({sum(len(t) for t in held):,} chars)")
    d, t = E.coverage("".join(train))
    print(f"decomposable Ethiopic characters in training text: {d}/{t} ({d/t*100:.1f}%)\n")

    print(f"{'vocab':>7}  {'RAW tok/char':>13} {'fert':>7}   "
          f"{'DECOMP tok/char':>16} {'fert':>7}   {'decomp vs raw':>14}")
    rows = []
    for v in BUDGETS:
        t_raw = train_bpe(train_raw, v)
        t_dec = train_bpe(train_dec, v)
        n_r, tc_r, f_r = measure(t_raw, held, decompose=False)
        n_d, tc_d, f_d = measure(t_dec, held, decompose=True)
        ratio = n_d / n_r
        verdict = "decomp wins" if ratio < 1 else "raw wins"
        print(f"{v:7d}  {tc_r:13.4f} {f_r:7.2f}   {tc_d:16.4f} {f_d:7.2f}   "
              f"{ratio:9.3f}x  {verdict}")
        rows.append({"vocab": v, "raw_tokens": n_r, "dec_tokens": n_d,
                     "raw_tok_per_char": tc_r, "dec_tok_per_char": tc_d,
                     "raw_fertility": f_r, "dec_fertility": f_d, "ratio": ratio})

    Path("data/decomposition_results.json").write_text(json.dumps(rows, indent=2))
    print("\nwrote data/decomposition_results.json")

    wins = [r["vocab"] for r in rows if r["ratio"] < 1]
    if wins:
        print(f"decomposition wins at vocab sizes: {wins}")
    else:
        print("decomposition loses at every budget tested")


if __name__ == "__main__":
    main()
