"""
Demo regression runner (Nov 18).

Plays evaluation/demo_script.json through a fresh DMGPipeline on a temporary
graph, checks every answer (expect / forbid keywords) and the final graph
(final_active / final_retired / final_absent), and exits non-zero on failure.

Run from the project root (Ollama must be running):
    python -m evaluation.run_demo
    python -m evaluation.run_demo --model qwen2.5:3b-instruct --runs 3

Helpers the UI can import:
    load_script(path=DEMO_SCRIPT_PATH) -> dict
    check_turn(answer, turn) -> (ok, reasons)
    check_graph(store, script) -> [(check, ok), ...]
    run_script(script, model=None, verbose=True) -> {"turns": [...], "graph": [...], "passed": int, "total": int}
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EVAL_DIR, MODEL_NAME
from main import DMGPipeline

DEMO_SCRIPT_PATH = os.path.join(EVAL_DIR, "demo_script.json")


def load_script(path=DEMO_SCRIPT_PATH):
    """Load the demo script JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_turn(answer, turn):
    """Check one answer against the turn's expect / forbid lists; return (ok, reasons)."""
    text = (answer or "").lower()
    reasons = [f"missing '{word}'" for word in turn.get("expect", []) if word.lower() not in text]
    reasons += [f"contains forbidden '{phrase}'" for phrase in turn.get("forbid", [])
                if phrase.lower() in text]
    return not reasons, reasons


def _matches(fact, keywords):
    """True if the fact text contains every keyword (case-insensitive)."""
    text = fact["text"].lower()
    return all(word.lower() in text for word in keywords)


def check_graph(store, script):
    """Run the final graph checks; return a list of (check description, ok)."""
    facts = store.get_all_facts(active_only=False)
    active = [fact for fact in facts if not fact.get("ended_at")]
    retired = [fact for fact in facts if fact.get("ended_at")]
    results = []
    for keywords in script.get("final_active", []):
        ok = any(_matches(fact, keywords) for fact in active)
        results.append((f"active:  {' + '.join(keywords)}", ok))
    for keywords in script.get("final_retired", []):
        ok = (any(_matches(fact, keywords) for fact in retired)
              and not any(_matches(fact, keywords) for fact in active))
        results.append((f"retired: {' + '.join(keywords)}", ok))
    for keywords in script.get("final_absent", []):
        ok = not any(_matches(fact, keywords) for fact in active)
        results.append((f"absent:  {' + '.join(keywords)}", ok))
    return results


def run_script(script, model=None, verbose=True):
    """Play the script through a fresh pipeline and return the per-check results."""
    tmp_dir = tempfile.mkdtemp(prefix="dmg_demo_")
    pipeline = DMGPipeline(os.path.join(tmp_dir, "graph.json"), model=model)
    pipeline.reset()
    turn_results = []
    try:
        for number, turn in enumerate(script["turns"], start=1):
            result = pipeline.chat(turn["message"])
            checked = "expect" in turn or "forbid" in turn
            ok, reasons = check_turn(result["answer"], turn)
            if checked:
                turn_results.append((f"turn {number:>2}: {turn['message']}", ok, reasons))
            if verbose:
                print(f"\n[{number}] You: {turn['message']}")
                if result["rewritten_query"] != turn["message"]:
                    print(f"     rewritten: {result['rewritten_query']}")
                for action in result["actions"]:
                    print(f"     [{action['op']}] {action['subject']} {action['predicate']} {action['object']}")
                print(f"     facts: {[fact['text'] for fact in result['retrieved_facts']]}")
                print(f"     Bot: {result['answer']}")
                print(f"     ({result['timings']['total']:.1f}s)", end="")
                print(f"  {'PASS' if ok else 'FAIL ' + '; '.join(reasons)}" if checked else "")
        graph_results = check_graph(pipeline.store, script)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    checks = [ok for _, ok, _ in turn_results] + [ok for _, ok in graph_results]
    if verbose:
        print("\nGraph checks:")
        for name, ok in graph_results:
            print(f"   {'PASS' if ok else 'FAIL'}  {name}")
        print(f"\nScore: {sum(checks)}/{len(checks)} "
              f"(turns {sum(ok for _, ok, _ in turn_results)}/{len(turn_results)}, "
              f"graph {sum(ok for _, ok in graph_results)}/{len(graph_results)})")
    return {"turns": turn_results, "graph": graph_results,
            "passed": sum(checks), "total": len(checks)}


def main():
    """Command-line entry point: run the script N times and summarise."""
    parser = argparse.ArgumentParser(description="Run the DMG demo script")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--quiet", action="store_true", help="only print summaries")
    args = parser.parse_args()

    script = load_script()
    passes, failures, total = Counter(), 0, 0
    for run in range(1, args.runs + 1):
        print(f"\n===== Run {run}/{args.runs} ({args.model}) =====")
        summary = run_script(script, model=args.model, verbose=not args.quiet)
        print(f"Run {run}: {summary['passed']}/{summary['total']}")
        for name, ok, *_ in summary["turns"] + summary["graph"]:
            passes[name] += ok
        failures += summary["total"] - summary["passed"]
        total = summary["total"]

    if args.runs > 1:
        print(f"\nPer-check passes over {args.runs} runs:")
        for name, count in passes.items():
            print(f"   {count}/{args.runs}  {name}")
    print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} failed checks'} "
          f"({args.runs} run(s) x {total} checks)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
