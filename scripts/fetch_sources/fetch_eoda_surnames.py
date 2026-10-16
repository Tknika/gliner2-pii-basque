#!/usr/bin/env python3
"""Surname candidates from EODA (Euskaltzaindia's onomastics database).

Pages through the `mota=deiturak` search ten results at a time until a
page brings nothing new.

    python scripts/fetch_sources/fetch_eoda_surnames.py [--delay 1.0]

Writes surnames_eod.json into --out. Only used as the candidate pool for
fetch_ine.py; the surname list itself is not redistributed.
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

URL = ("https://www.euskaltzaindia.eus/index.php?"
       "option=com_ecoeoda&Itemid=469&task=bilaketa&view=bilaketa"
       "&lang=eu&query=%2A%3A%2A&mota=deiturak&ordena=score&nondik={start}")
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Firefox/128.0"}
TITLE = re.compile(r"<h3>\s*<a[^>]*>\s*([^<]+?)\s*</a>", re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=1.0,
                    help="seconds between requests")
    ap.add_argument("--max-pages", type=int, default=4000)
    ap.add_argument("--out", type=Path, default=Path("data/sources"))
    args = ap.parse_args()

    surnames, seen = [], set()
    for page in range(args.max_pages):
        request = urllib.request.Request(URL.format(start=page * 10),
                                         headers=HEADERS)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=30) as r:
                    html = r.read().decode("utf-8", errors="ignore")
                break
            except Exception as e:  # network errors: retry
                print(f"page {page}: {e}", file=sys.stderr)
                time.sleep(10)
        else:
            raise SystemExit("too many errors")
        new = [s for s in (m.strip() for m in TITLE.findall(html))
               if s not in seen]
        if not new:
            break
        seen.update(new)
        surnames.extend(new)
        time.sleep(args.delay)

    path = args.out / "surnames_eod.json"
    path.write_text(json.dumps(surnames, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"✓ {len(surnames)} surnames → {path}")


if __name__ == "__main__":
    main()
