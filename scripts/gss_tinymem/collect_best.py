"""Select the final GSS configuration per TinyMem setting and write
configs/gss_tinymem_best.json and analysis/gss_tinymem_search/SUMMARY.md.

Rule (decided 2026-10-20): whitened construction only, memorization exactly 0, then the
best utility (accuracy for math, perplexity for LM).
  protocol "paper"  : stage-B rows are already measured on the test metrics (paper metric:
                      all noised rows per duplication class / poisoned-test ASR; clean test
                      loaders), so the selected row's numbers are the reported numbers.
  protocol "heldout": selection on the validation splits; test numbers are shown only when a
                      stage-C row exists for exactly that configuration.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

OUT = os.path.join(C.REPO, "results", "gss_tinymem_search")
KIND = "whitened"
PAPER = {  # Table 1: Ours (%Mem / utility) and Baseline
    "math_noise": ("0.00 / Acc 96.98", "34.55 / Acc 97.00"),
    "math_backdoor": ("0.00 / Acc 96.82", "99.44 / Acc 96.81"),
    "lm_noise": ("0.00 / PPL 63.13", "17.60 / PPL 57.73"),
    "lm_backdoor": ("0.00 / PPL 63.17", "100.00 / PPL 58.45"),
}
CFG_KEYS = ("point", "kind", "tau", "rank", "q", "beta", "sign", "r")


def load_rows(name):
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, f"{name}*.jsonl"))):
        if "_smoke" in p:
            continue
        for line in open(p):  # several search passes (tags) may exist; pick across all of them
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            r.setdefault("protocol", "heldout")
            rows.append(r)
    return rows


def util_str(lm, acc, ppl):
    return f"PPL {ppl:.2f}" if lm else f"Acc {100 * acc:.2f}"


def describe(r):
    return (f"ref={r['ref']} {r['point'].replace('transformer.', '')} tau={r['tau']:g} K={r['rank']} "
            f"q={r['q']} beta={r['beta']:g} sign={r['sign']:+g} r={r['r']}")


best = {"paper": {}, "heldout": {}}
lines = ["# GSS on TinyMem: final configurations", "",
         f"Construction: {KIND} only. Selection: memorization exactly 0, then best utility. "
         "The steering hook is active for every metric.", ""]
for protocol in ("paper", "heldout"):
    if protocol == "paper":
        lines += ["## Protocol `paper` (hyper-parameters tuned on the test metrics)", "",
                  "| setting | paper Ours | baseline (ours) | GSS mem | GSS utility | gate on clean test tokens | config |",
                  "|---|---|---|---|---|---|---|"]
    else:
        lines += ["## Protocol `heldout` (selected on validation; test only where already evaluated)", "",
                  "| setting | val mem | val utility | test mem (held-out) | test utility | config |",
                  "|---|---|---|---|---|---|"]
    for name in C.SETTINGS:
        lm = name.startswith("lm")
        rows = [r for r in load_rows(name) if r["protocol"] == protocol]
        base = next((r for r in rows if r.get("stage") == "base"), None)
        zero = [r for r in rows if r.get("stage") == "B" and r["kind"] == KIND and r["val_mem"] == 0]
        if not zero:
            lines.append(f"| {name} | – | (no zero-memorization {KIND} configuration) | | | | |")
            continue
        sel = min(zero, key=lambda r: r["val_ppl"]) if lm else max(zero, key=lambda r: r["val_acc"])
        cfg = {k: sel[k] for k in CFG_KEYS}
        cfg["gamma_rel"] = 1e-3
        entry = dict(ref=sel["ref"], config=cfg, eps=sel["eps"], selected_from=sel["key"],
                     n_zero=len(zero), n_rows=sum(r.get("stage") == "B" and r["kind"] == KIND for r in rows))
        if protocol == "paper":
            entry.update(test_mem=sel["val_mem"], test_acc=sel["val_acc"], test_ppl=sel["val_ppl"],
                         gate_clean_test=sel["gate_clean"])
            best[protocol][name] = entry
            b = util_str(lm, base["val_acc"], base["val_ppl"]) if base else "–"
            bm = f"{100 * base['val_mem']:.2f}" if base else "–"
            lines.append(f"| {name} | {PAPER[name][0]} | {bm} / {b} | 0.00 | {util_str(lm, sel['val_acc'], sel['val_ppl'])} "
                         f"| {100 * sel['gate_clean']:.3f}% | {describe(sel)} |")
        else:
            c = next((r for r in rows if r.get("stage") == "C" and r["ref"] == sel["ref"]
                      and all(r["config"][k] == sel[k] for k in CFG_KEYS)), None)
            if c:
                entry.update(test_mem=c["test_mem"], test_acc=c["test_acc"], test_ppl=c["test_ppl"])
            best[protocol][name] = entry
            tm = f"{100 * c['test_mem']:.2f}" if c else "not evaluated"
            tu = util_str(lm, c["test_acc"], c["test_ppl"]) if c else "–"
            lines.append(f"| {name} | 0.00 | {util_str(lm, sel['val_acc'], sel['val_ppl'])} | {tm} | {tu} | {describe(sel)} |")
    lines.append("")
lines += ["Memorization: noise settings = % of noised rows reproduced (max over duplication classes); "
          "backdoor = attack success on the poisoned test set (paper) / poisoned held-out train rows (heldout val)."]
os.makedirs(os.path.join(C.REPO, "configs"), exist_ok=True)
with open(os.path.join(C.REPO, "configs", "gss_tinymem_best.json"), "w") as f:
    json.dump(best, f, indent=1)
open(os.path.join(OUT, "SUMMARY.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
