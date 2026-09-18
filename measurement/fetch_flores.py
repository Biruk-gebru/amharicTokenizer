"""Fetch FLORES+ dev sentences for Amharic, English and Korean.

FLORES+ is a second, independent parallel corpus. UDHR alone is one document in one
register, which is the weakest point of a single-corpus benchmark; FLORES+ is 997
professionally translated sentences aligned across 200+ languages and is the standard
corpus in this literature.

LICENSE OBLIGATION -- read before changing this file.
Access requires agreeing to two terms on the Hub:

  1. "I agree not to re-host FLORES+ in places where it could be picked up by web
     crawlers."
  2. "If I evaluate using FLORES+, I will ensure that its contents are not in the
     training data."

So FLORES+ is **fetched, never committed**. `data/flores/` is gitignored. Do not add it
to the repo, and do not use it to train anything -- it is held-out evaluation only. The
BPE training in decomposition_experiment.py uses Amharic Wikipedia for exactly this
reason.

Requires a Hugging Face token with the dataset's terms accepted.
"""
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "openlanguagedata/flores_plus"
LANGS = {"amh": "amh_Ethi", "eng": "eng_Latn", "kor": "kor_Hang"}
OUT = Path("data/flores")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for short, code in LANGS.items():
        try:
            src = hf_hub_download(REPO, f"dev/{code}.jsonl", repo_type="dataset")
        except Exception as e:
            sys.exit(f"cannot fetch {code}: {type(e).__name__}: {e}\n"
                     f"Accept the terms at https://huggingface.co/datasets/{REPO}")
        rows = [json.loads(l) for l in Path(src).read_text(encoding="utf-8").splitlines() if l.strip()]
        # Keep sentence order identical across languages so the arms stay aligned.
        rows.sort(key=lambda r: r.get("id", 0))
        text = "\n".join(r["text"] for r in rows)
        (OUT / f"{short}.txt").write_text(text, encoding="utf-8")
        eth = sum(1 for c in text if 0x1200 <= ord(c) < 0x1380)
        print(f"  {short}: {len(rows)} sentences, {len(text):,} chars"
              + (f", {eth/len(text)*100:.1f}% Ethiopic" if eth else ""))

    lens = {s: len((OUT / f"{s}.txt").read_text(encoding="utf-8").splitlines()) for s in LANGS}
    assert len(set(lens.values())) == 1, f"sentence counts differ across languages: {lens}"
    print(f"\nwrote {OUT}/ -- {list(lens.values())[0]} aligned sentences per language")
    print("NOT committed: data/flores/ is gitignored per the FLORES+ re-hosting term")


if __name__ == "__main__":
    main()
