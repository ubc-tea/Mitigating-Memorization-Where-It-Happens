"""Check that the teacher-forced memorization metric equals the generation-based one,
row by row, with and without a steering hook, on every TinyMem setting."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import torch  # noqa: E402

from src import gss_tinymem as gss  # noqa: E402

out = []
for name in sys.argv[1:] or list(C.SETTINGS):
    S = C.Setting(name)
    ref = C.load_model(S.cfg["refs"][list(S.cfg["refs"])[0]], S.cfg["vocab"])
    cfgs = [None,
            gss.GSSConfig(point="transformer.h.3.mlp", kind="whitened", q=0.99, beta=1.0, sign=-1.0),
            gss.GSSConfig(point="transformer.h.2", kind="contrastive", q=0.95, beta=10.0, sign=1.0, r=2)]
    for cfg in cfgs:
        hook = None
        if cfg is not None:
            hook = gss.attach(S.model, gss.fit(S.model, ref, S.calib_rows, S.clean_eps_rows, cfg))
        try:
            if S.cfg["backdoor"]:
                P, Cl = S.test_poison, S.test_clean
                slow = C.backdoor_success(S.model, P, Cl, S.d["trigger"], S.cfg["data_name"])
                fast = C.backdoor_success_fast(S.model, P, Cl, S.d["trigger"], S.cfg["data_name"])
            else:
                slow, _ = C.noise_memorized(S.model, S.noise, S.clean)
                fast, _ = C.noise_memorized_fast(S.model, S.noise, S.clean)
        finally:
            if hook:
                hook.remove()
        rec = dict(setting=name, config=cfg.to_dict() if cfg else None, n=int(slow.numel()),
                   slow=int(slow.sum()), fast=int(fast.sum()),
                   mismatches=int((slow.cpu() != fast.cpu()).sum()))
        out.append(rec)
        print(json.dumps(rec), flush=True)
