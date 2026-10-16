#!/usr/bin/env python3
"""Audit synthetic JSONL files (checksums, formats, spans, distributions).

    python scripts/audit_data.py data/synthetic/*.jsonl

Exits with status 1 if any check fails.
"""

import sys

from gliner2_pii_basque.synthetic.audit import audit
from gliner2_pii_basque.synthetic.generator import read_jsonl

status = 0
for path in sys.argv[1:]:
    errors, report = audit(read_jsonl(path))
    print(f"── {path}")
    print("\n".join(report))
    if errors:
        status = 1
        print("ERRORS:")
        for kind, count in errors.most_common():
            print(f"  {kind}: {count}")
    else:
        print("✓ all checks pass")
    print()
sys.exit(status)
