"""Fit GSS on one TinyMem setting with a given configuration and report test metrics.

    python scripts/gss_tinymem/run_gss_tinymem.py --setting math_backdoor            # best config
    python scripts/gss_tinymem/run_gss_tinymem.py --setting math_noise \
        --config '{"point": "transformer.h.3", "kind": "contrastive", "q": 0.95, "beta": 1}' --ref xbd

The steering hook is active for every reported metric. Directions are fitted on the
calibration split only; see common.py for the splits.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import torch  # noqa: E402

from src import gss_tinymem as gss  # noqa: E402

BEST = os.path.join(C.REPO, "configs", "gss_tinymem_best.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True, choices=list(C.SETTINGS))
    ap.add_argument("--config", default=None, help="JSON GSSConfig fields; default: configs/gss_tinymem_best.json")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--protocol", default="paper", choices=["paper", "heldout"],
                    help="which selection of configs/gss_tinymem_best.json, and which calibration split")
    ap.add_argument("--save", default=None, help="where to save the fitted artifact (.pt)")
    a = ap.parse_args()
    if a.config:
        cfg, ref = gss.GSSConfig(**json.loads(a.config)), a.ref
    else:
        best = json.load(open(BEST))[a.protocol][a.setting]
        cfg, ref = gss.GSSConfig(**best["config"]), best["ref"]
    S = C.Setting(a.setting, protocol=a.protocol)
    ref = ref or list(S.cfg["refs"])[0]
    t0 = time.time()
    art = gss.fit(S.model, C.load_model(S.cfg["refs"][ref], S.cfg["vocab"]), S.calib_rows, S.clean_eps_rows, cfg)
    fit_sec = time.time() - t0
    art.update(ref=ref, setting=a.setting)
    if a.save:
        torch.save(art, a.save)
    base = dict(test_mem=S.mem(S.model, "test"), paper_mem=S.paper_mem(S.model))
    base["test_acc"], base["test_ppl"] = S.utility(S.model, "test")
    hook = gss.attach(S.model, art)
    try:
        res = dict(test_mem=S.mem(S.model, "test"), paper_mem=S.paper_mem(S.model))
        hook.reset_counts()
        res["test_acc"], res["test_ppl"] = S.utility(S.model, "test")
        res["gate_clean_test"] = hook.rate
    finally:
        hook.remove()
    print(json.dumps(dict(setting=a.setting, protocol=a.protocol, ref=ref, config=cfg.to_dict(), eps=art["eps"], fit_sec=fit_sec,
                          baseline=base, gss=res), indent=1))


if __name__ == "__main__":
    main()
