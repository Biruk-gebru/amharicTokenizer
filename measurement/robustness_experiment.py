"""Stress-test decomposition_experiment.py before anyone quotes it.

    python robustness_experiment.py am
    python robustness_experiment.py ko

decomposition_experiment.py trains one byte-level BPE per arm on one shuffle of raw
Wikipedia and evaluates on held-out paragraphs from the same dump. Four ways that could
be unfair to decomposition, each addressed here:

  data       the dump has duplicate paragraphs, "==" heading residue and ~9% Latin
             (am), and some held-out paragraphs also appear in train. Cleaned and
             deduplicated here, with exact-duplicate leakage removed.
  domain     held-out Wikipedia shares templates with train. FLORES+ (devtest-style,
             professionally translated, never trained on) is the main eval here.
  model      byte-level BPE always starts from 256 bytes, so decomposition cannot save
             alphabet slots, which is the RFC's core argument. Character-level BPE and
             Unigram (what KR-BERT and SentencePiece use) are added, and for those the
             rate of unknown tokens is reported as well as length.
  variance   three shuffles instead of one.

Ethiopic gets a second decomposition, "lean": the 6th-order (bare consonant) form stays
one character and only the other six orders split. It is shorter, at the cost of the
rare base+vowel collisions ethiopic.collision_rate() counts.
"""

import json, random, statistics, sys
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

BUDGETS = [1000, 2000, 4000, 8000, 16000, 32000]
SEEDS = [20260918, 1, 2]
LANG = sys.argv[1] if len(sys.argv) > 1 else "am"

if LANG == "am":
    import ethiopic as S
    in_script = lambda c: 0x1200 <= ord(c) < 0x1380
    LEAN = {k: v for k, v in S.DECOMPOSE.items() if k not in S.BASES}
    ARMS = {
        "raw": lambda t: t,
        "decomp": S.decompose,
        "lean": lambda t: "".join("".join(LEAN[c]) if c in LEAN else c for c in t),
    }
    FLORES = "amh"
elif LANG == "ko":
    import hangul as S
    in_script = lambda c: 0xAC00 <= ord(c) < 0xD7A4
    ARMS = {"raw": lambda t: t, "decomp": lambda t: S.decompose(t, "compat")}
    FLORES = "kor"
else:
    sys.exit(f"unknown language {LANG!r}; use 'am' or 'ko'")


def clean_corpus():
    paras = [p.strip() for p in Path(f"data/{LANG}_wiki.txt").read_text("utf-8").split("\n\n")]
    seen, out = set(), []
    for p in paras:
        letters = [c for c in p if c.isalpha()]
        if len(p) < 40 or p.startswith("==") or p in seen:
            continue
        if sum(map(in_script, letters)) / max(len(letters), 1) < 0.8:
            continue
        seen.add(p)
        out.append(p)
    return out


def build(kind, vocab):
    if kind == "bytebpe":
        tok = Tokenizer(models.BPE(unk_token=None))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tr = trainers.BpeTrainer(vocab_size=vocab, show_progress=False,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    elif kind == "charbpe":
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.Metaspace()
        tr = trainers.BpeTrainer(vocab_size=vocab, show_progress=False,
                                 special_tokens=["<unk>"])
    else:
        tok = Tokenizer(models.Unigram())
        tok.pre_tokenizer = pre_tokenizers.Metaspace()
        tr = trainers.UnigramTrainer(vocab_size=vocab, show_progress=False,
                                     special_tokens=["<unk>"], unk_token="<unk>")
    return tok, tr


def measure(tok, texts, fn):
    n_tok = n_unk = 0
    unk = tok.token_to_id("<unk>")
    for t in texts:
        ids = tok.encode(fn(t), add_special_tokens=False).ids
        n_tok += len(ids)
        n_unk += ids.count(unk) if unk is not None else 0
    return n_tok, n_unk


def main():
    paras = clean_corpus()
    flores = Path(f"data/flores/{FLORES}.txt").read_text("utf-8").splitlines()
    print(f"{LANG}: {len(paras)} clean paragraphs, {sum(map(len, paras)):,} chars; "
          f"FLORES+ {len(flores)} sentences, {sum(map(len, flores)):,} chars")
    rows = []
    for seed in SEEDS:
        random.Random(seed).shuffle(paras)
        cut = int(len(paras) * 0.9)
        train, held = paras[:cut], paras[cut:]   # deduplicated, so no exact leakage
        for kind in ["bytebpe", "charbpe", "unigram"]:
            for v in BUDGETS:
                for arm, fn in ARMS.items():
                    tok, tr = build(kind, v)
                    try:
                        tok.train_from_iterator([fn(t) for t in train], trainer=tr)
                    except Exception as e:   # Unigram: alphabet larger than the budget
                        print(f"  seed {seed} {kind} {v} {arm}: {e}")
                        continue
                    for ev, texts in [("wiki", held), ("flores", flores)]:
                        n, u = measure(tok, texts, fn)
                        rows.append({"seed": seed, "model": kind, "vocab": v, "arm": arm,
                                     "eval": ev, "tokens": n, "unk": u,
                                     "chars": sum(map(len, texts)),
                                     "actual_vocab": tok.get_vocab_size()})
                print(f"  seed {seed} {kind} {v} done", flush=True)

    out = Path(f"data/robustness_results_{LANG}.json")
    out.write_text(json.dumps(rows, indent=1))
    summarize(rows)
    print(f"\nwrote {out}")


def summarize(rows):
    get = lambda **k: [r for r in rows if all(r[a] == b for a, b in k.items())]
    for ev in ["flores", "wiki"]:
        print(f"\n== eval: {ev}   ratio = arm tokens / raw tokens, mean [min-max] over "
              f"{len(SEEDS)} seeds; unk = % of tokens that are <unk>")
        for kind in ["bytebpe", "charbpe", "unigram"]:
            print(f"-- {kind}")
            for v in BUDGETS:
                line = f"  {v:6d}"
                raw = {r["seed"]: r for r in get(model=kind, vocab=v, arm="raw", eval=ev)}
                if not raw:
                    print(f"  {v:6d}  raw could not be trained (alphabet exceeds budget)")
                    continue
                if max(r["actual_vocab"] for r in raw.values()) > v:   # alphabet > budget
                    line += "  RAW OVER BUDGET, not comparable"
                line += f"  raw unk {statistics.mean(r['unk'] / r['tokens'] for r in raw.values()) * 100:5.2f}%"
                for arm in [a for a in ARMS if a != "raw"]:
                    rs = [r for r in get(model=kind, vocab=v, arm=arm, eval=ev)
                          if r["seed"] in raw]
                    if not rs:
                        line += f"   {arm} untrainable"
                        continue
                    ratios = [r["tokens"] / raw[r["seed"]]["tokens"] for r in rs]
                    unk = statistics.mean(r["unk"] / r["tokens"] for r in rs) * 100
                    line += (f"   {arm} {statistics.mean(ratios):.3f}x "
                             f"[{min(ratios):.3f}-{max(ratios):.3f}] unk {unk:5.2f}%")
                print(line)


if __name__ == "__main__":
    main()
