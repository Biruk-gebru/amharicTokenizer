"""Fetch an Amharic training corpus from Amharic Wikipedia (CC BY-SA 4.0).

UDHR is ~5k characters, far too small to train a BPE vocabulary on. This walks
am.wikipedia.org and saves article extracts. No account or token needed.

Two things learned the hard way and encoded here:
  * Wikimedia returns 403 to any request without a descriptive User-Agent. That is
    published policy, not rate limiting -- a bare urllib request always fails.
  * `generator=random` re-draws pages it has already returned, so throughput collapses
    on a small wiki. Walking `allpages` with continuation visits each page once.
  * `exlimit` caps at 20, and for WHOLE-article extracts MediaWiki silently lowers it
    to 1 -- one article per request. `exintro` lifts that to 20 per request, so we take
    article intros. Amharic Wikipedia is mostly short articles, so little is lost.
  * The continuation key is not always `gapcontinue` (it can be `excontinue`), so the
    whole `continue` object is echoed back rather than one hand-picked field.

Progress is flushed to disk as it goes, so an interrupted run still leaves a usable
corpus instead of nothing.
"""
import json, sys, time, urllib.parse, urllib.request
from pathlib import Path

API = "https://am.wikipedia.org/w/api.php"
UA = {"User-Agent": "amharicTokenizer-research/0.1 "
                    "(https://github.com/Biruk-gebru/amharicTokenizer)"}
OUT = Path("data/amh_wiki.txt")
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000


def get(params):
    q = urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    req = urllib.request.Request(f"{API}?{q}", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main():
    OUT.parent.mkdir(exist_ok=True)
    total, n, cont = 0, 0, {}
    with OUT.open("w", encoding="utf-8") as fh:
        while total < TARGET:
            params = {"action": "query", "generator": "allpages", "gapnamespace": 0,
                      "gaplimit": 20, "prop": "extracts", "explaintext": 1,
                      "exintro": 1, "exlimit": 20, **cont}
            try:
                data = get(params)
            except Exception as e:
                print(f"  retry after {type(e).__name__} {getattr(e,'code','')}", flush=True)
                time.sleep(3)
                continue

            for p in data.get("query", {}).get("pages", []):
                t = p.get("extract", "").strip()
                if len(t) > 80:
                    fh.write(t + "\n\n")
                    total += len(t)
                    n += 1
            fh.flush()
            print(f"  {n} articles, {total:,} chars", flush=True)

            nxt = data.get("continue")
            if not nxt:
                print("  reached end of wiki")
                break
            cont = nxt
            time.sleep(0.3)

    text = OUT.read_text(encoding="utf-8")
    eth = sum(1 for c in text if 0x1200 <= ord(c) < 0x1380)
    print(f"\nwrote {OUT}: {len(text):,} chars, {eth/max(len(text),1)*100:.1f}% Ethiopic, "
          f"{n} articles")


if __name__ == "__main__":
    main()
