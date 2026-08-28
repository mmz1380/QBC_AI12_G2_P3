"""Project startup script — one entry point for the whole pipeline.

Usage:
    python run.py setup                 # download the raw dataset
    python run.py clean [--sample]      # Phase 1: clean (full, or notebook sample)
    python run.py eda                   # Phase 1: EDA figures
    python run.py index                 # Phase 2: build the product embedding index
    python run.py train                 # Phase 3: train the recommendation classifier
    python run.py eval                  # Phase 4: run the evaluation suite
    python run.py lora                  # bonus: LoRA fine-tune vs. the TF-IDF baseline
    python run.py dashboard             # launch the Streamlit dashboard
    python run.py all [--sample]        # setup -> clean -> eda -> index -> train -> eval
    python run.py menu                  # interactive menu

Run mode / data volume are controlled by env vars (see src/digikala/config.py):
    DIGIKALA_RUN_MODE=local|free|paid   DIGIKALA_SAMPLE=full|<int>
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def _log():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def cmd_setup(_):
    from digikala.core import dataio
    dataio.download_raw()


def cmd_clean(args):
    from digikala.phase1_data import clean
    rep = clean.build(full=not args.sample)
    print("products:", rep["products"].get("output_rows"),
          "| comments:", rep["comments"].get("output_rows"))


def cmd_eda(_):
    from digikala.phase1_data import eda
    print(eda.run())


def cmd_index(_):
    from digikala.phase2_assistant import retrieval
    retrieval.build_product_index()


def cmd_train(_):
    from digikala.phase3_predict import recommend
    recommend.train_and_save()


def cmd_eval(_):
    from digikala.phase4_eval import evaluate
    evaluate.run()


def cmd_lora(_):
    from digikala.phase3_predict import lora_finetune
    print(lora_finetune.train_and_compare())


def cmd_demo(args):
    import json
    from digikala import demo
    print(json.dumps(demo.run(sample_size=args.sample_size), ensure_ascii=False, indent=2))


def cmd_test(_):
    subprocess.run([sys.executable, "-m", "pytest", "-q",
                    str(Path(__file__).resolve().parent / "tests")], check=False)


def cmd_dashboard(_):
    app = Path(__file__).resolve().parent / "dashboard" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)], check=False)


def cmd_all(args):
    cmd_setup(args)
    cmd_clean(args)
    cmd_eda(args)
    cmd_index(args)
    cmd_train(args)
    cmd_eval(args)


def cmd_menu(_):
    items = [("setup", cmd_setup), ("clean (sample)", lambda a: cmd_clean(argparse.Namespace(sample=True))),
             ("clean (full)", lambda a: cmd_clean(argparse.Namespace(sample=False))),
             ("eda", cmd_eda), ("index", cmd_index), ("train", cmd_train),
             ("eval", cmd_eval), ("dashboard", cmd_dashboard)]
    print("\nDigikala Project 3 — pick a step:")
    for i, (name, _) in enumerate(items, 1):
        print(f"  {i}) {name}")
    choice = input("number> ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(items):
        items[int(choice) - 1][1](argparse.Namespace())


def main():
    _log()
    p = argparse.ArgumentParser(description="Digikala Project 3 pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, has_sample in [
        ("setup", cmd_setup, False), ("clean", cmd_clean, True), ("eda", cmd_eda, False),
        ("index", cmd_index, False), ("train", cmd_train, False), ("eval", cmd_eval, False),
        ("dashboard", cmd_dashboard, False), ("all", cmd_all, True), ("menu", cmd_menu, False),
        ("demo", cmd_demo, False), ("test", cmd_test, False), ("lora", cmd_lora, False)]:
        sp = sub.add_parser(name)
        if has_sample:
            sp.add_argument("--sample", action="store_true",
                            help="use the notebook-sized comments sample instead of the full 6M rows")
        if name == "demo":
            sp.add_argument("--sample-size", type=int, default=20_000,
                            help="comments in the deterministic demo sample (must match the notebook)")
        sp.set_defaults(func=fn)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
