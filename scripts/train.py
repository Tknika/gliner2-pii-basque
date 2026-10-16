#!/usr/bin/env python3
"""Fine-tune GLiNER2-PII on Basque NER, optionally with PII replay.

Training data: BasqueGLUE nerc_id/train (2,842 sentences) + an optional
synthetic replay file. Checkpoint selection: lowest loss on
nerc_id/validation. Test splits are never touched here; run
scripts/evaluate.py on the result.

Recipe of tknika/gliner2-PII-basque-v2:

    python scripts/train.py --mode lora \\
        --replay-file data/synthetic/v2-train.jsonl --output-dir outputs/v2
    python scripts/merge_adapter.py outputs/v2/best outputs/v2/merged

Smoke test (a few steps):  add --max-steps 5
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import torch
from gliner2 import AutoExtractor
from gliner2.training.data import InputExample
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

from gliner2_pii_basque import BASE_MODEL, basqueglue
from gliner2_pii_basque.synthetic.generator import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def to_examples(records):
    return [InputExample(text=r["text"], entities=r["entities"])
            for r in records]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["full", "lora"], default="lora")
    ap.add_argument("--replay-file", type=Path,
                    help="synthetic PII JSONL mixed into the training data")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--encoder-lr", type=float, default=1e-5)
    ap.add_argument("--task-lr", type=float, default=5e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
          else "none")
    train = basqueglue.load("nerc_id/train")
    validation = basqueglue.load("nerc_id/validation")
    print(f"nerc_id/train: {len(train)}  nerc_id/validation: {len(validation)}")
    replay = read_jsonl(args.replay_file) if args.replay_file else []
    if replay:
        print(f"replay ({args.replay_file}): {len(replay)}")

    config = TrainingConfig(
        output_dir=str(args.output_dir),
        experiment_name=f"gliner2-pii-basque-{args.mode}"
                        f"{'-replay' if replay else ''}",
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        encoder_lr=args.encoder_lr,
        task_lr=args.task_lr,
        bf16=torch.cuda.is_available(),
        fp16=False,
        eval_strategy="epoch",
        metric_for_best="eval_loss",
        save_best=True,
        save_total_limit=1,
        logging_steps=20,
        validate_data=True,
        seed=args.seed,
        use_lora=args.mode == "lora",
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        save_adapter_only=False,
    )
    model = AutoExtractor.from_pretrained(args.base_model)
    trainer = GLiNER2Trainer(model, config)
    start = time.time()
    summary = trainer.train(train_data=to_examples(train + replay),
                            eval_data=to_examples(validation))

    run = {
        "args": {k: str(v) if isinstance(v, Path) else v
                 for k, v in vars(args).items()},
        "train_sentences": {"nerc_id/train": len(train), "replay": len(replay)},
        "replay_sha256": sha256(args.replay_file) if replay else None,
        "seconds": round(time.time() - start),
        "summary": {k: v for k, v in summary.items()
                    if isinstance(v, (int, float, str))},
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "packages": {p: version(p) for p in
                     ("gliner2", "peft", "torch", "transformers")},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available()
               else None,
    }
    path = args.output_dir / "run.json"
    path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"done in {run['seconds']} s → {args.output_dir} ({path.name})")


if __name__ == "__main__":
    main()
