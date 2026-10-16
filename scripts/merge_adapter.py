#!/usr/bin/env python3
"""Merge a LoRA adapter into the base model → standalone model.

    python scripts/merge_adapter.py outputs/v2/best outputs/v2/merged

The merged model is reloaded and compared with the adapter model on a
probe sentence; the script fails if their predictions differ.
"""

import argparse

from gliner2 import AutoExtractor
from peft import PeftModel

from gliner2_pii_basque import BASE_MODEL
from gliner2_pii_basque.synthetic.templates import ENTITY_TYPES

PROBE = ("Kaixo, Ane Azkune naiz, Bilboko La Casilla kalean bizi naiz, 48002. "
         "Nire NAN-a 72154236Z da eta telefonoa 688 123 456. "
         "Posta: ane.azkune@euskaltel.eus.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter")
    ap.add_argument("output")
    ap.add_argument("--base-model", default=BASE_MODEL)
    args = ap.parse_args()

    adapted = PeftModel.from_pretrained(
        AutoExtractor.from_pretrained(args.base_model), args.adapter)
    expected = adapted.extract_entities(PROBE, ENTITY_TYPES)
    adapted.merge_and_unload().save_pretrained(args.output)

    reloaded = AutoExtractor.from_pretrained(args.output)
    got = reloaded.extract_entities(PROBE, ENTITY_TYPES)
    if got != expected:
        raise SystemExit(f"merged model differs from adapter model:\n"
                         f"  adapter: {expected}\n  merged:  {got}")
    print(f"✓ merged model saved and verified: {args.output}")


if __name__ == "__main__":
    main()
