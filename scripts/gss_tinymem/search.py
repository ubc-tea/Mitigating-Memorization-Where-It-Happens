"""Overnight search for the best honest GSS configuration on one TinyMem setting.

Stage A (no generation): for every (reference, tau, point, kind, rank) build the
  directions on calibration rows and measure gate selectivity on validation rows:
  fraction of memorized rows in which the gate fires right before a memorized token,
  versus the firing rate on held-out clean rows.  Keep the top --top_a configurations.
Stage B (generation, validation only): sweep q, beta, sign, r for the kept
  configurations; record validation memorization and utility.
  Selection: among configs with val memorization <= --mem_tol, best val utility
  (accuracy for math, perplexity for LM); otherwise the lowest val memorization.
Stage C: evaluate the selected configuration once on the test split and with the
  paper's metric; save the artifact and the config.
Every result line carries a key, so an interrupted run resumes where it stopped.
"""
import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import torch  # noqa: E402

from src import gss_tinymem as gss  # noqa: E402

OUT = os.path.join(C.REPO, "results", "gss_tinymem_search")


@torch.no_grad()
def fire_matrix(model, point, u, kind, eps, rows, r=1, bs=64):
    out = []

    def rec(_, __, o):
        s = gss._hidden(o).float() @ u
        ex = (s.abs() > eps) if kind == "whitened" else (s > eps)
        if r > 1:
            e = ex[..., 0].float()
            run = torch.zeros_like(e)
            for t in range(e.shape[1]):
                run[:, t] = (run[:, t - 1] + 1) * e[:, t] if t else e[:, t]
            out.append((run >= r).cpu())
        else:
            out.append(ex.any(-1).cpu())
        return o

    h = gss._module(model, point).register_forward_hook(rec)
    try:
        for i in range(0, rows.shape[0], bs):
            model.transformer(rows[i:i + bs])
    finally:
        h.remove()
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True)
    ap.add_argument("--refs", default=None)
    ap.add_argument("--taus", default="0,1,3")
    ap.add_argument("--points", default=",".join(
        [f"transformer.h.{i}" for i in range(4)] + [f"transformer.h.{i}.mlp" for i in range(4)]))
    ap.add_argument("--kinds", default="whitened", help="whitened (default, chosen 2026-10-20) or whitened,contrastive")
    ap.add_argument("--ranks", default="1,2,4")
    ap.add_argument("--qa", default="0.99,0.995,0.999")
    ap.add_argument("--top_a", type=int, default=6)
    ap.add_argument("--qb", default="0.95,0.98,0.99,0.995,0.999,0.9995")
    ap.add_argument("--betas", default="0.3,1,3,10")
    ap.add_argument("--signs", default="1,-1")
    ap.add_argument("--rs", default="1,2")
    ap.add_argument("--mem_tol", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--protocol", default="heldout", choices=["heldout", "paper"],
                    help="paper: select on the paper metric (all noised rows / poisoned test ASR) and "
                         "the clean test loaders, i.e. hyper-parameters are tuned on the test set")
    ap.add_argument("--per_kind", type=int, default=1, help="split top_a evenly across kinds")
    ap.add_argument("--seed_cfgs", default="[]", help="JSON list of [ref, tau, point, kind, rank] always swept")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    res_path = os.path.join(OUT, f"{a.setting}{a.tag}.jsonl")
    done = {}
    if os.path.exists(res_path):
        for line in open(res_path):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # line cut by a killed job
                continue
            done[rec["key"]] = rec

    def emit(rec):
        rec["setting"] = a.setting
        done[rec["key"]] = rec
        print("[RESULT]", json.dumps(rec), flush=True)
        with open(res_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    S = C.Setting(a.setting, protocol=a.protocol)
    paper = a.protocol == "paper"

    def sel_mem(model):  # memorization used for selection
        if paper:
            return max(S.paper_mem(model))
        return S.mem(model, "val")

    def sel_util(model):
        return S.utility(model, "test" if paper else "val")
    lm = S.cfg["vocab"] > 1000
    refs = a.refs.split(",") if a.refs else list(S.cfg["refs"])
    taus = [float(x) for x in a.taus.split(",")]
    points, kinds = a.points.split(","), a.kinds.split(",")
    ranks = [int(x) for x in a.ranks.split(",")]
    qa = [float(x) for x in a.qa.split(",")]

    if "baseline" not in done:
        t0 = time.time()
        acc, ppl = sel_util(S.model)
        emit(dict(key="baseline", stage="base", protocol=a.protocol, val_mem=sel_mem(S.model), val_acc=acc, val_ppl=ppl,
                  sec=time.time() - t0))

    P_val, T_val = S.target_positions("test" if (paper and S.cfg["backdoor"]) else "val")
    if paper:
        n_tasks = 5 if S.cfg["data_name"] == "mult" else 1
        clean_val_rows = torch.cat([dl.dataset for dl in S.d["clean_test_dataloaders"][:n_tasks]]).to(C.DEV)
    else:
        clean_val_rows = torch.cat(S.clean_val)
    ref_models = {}

    def get_ref(name):
        if name not in ref_models:
            ref_models.clear()
            torch.cuda.empty_cache()
            ref_models[name] = C.load_model(S.cfg["refs"][name], S.cfg["vocab"])
        return ref_models[name]

    def fit_dirs(ref, tau, point, kind, rank, cache={}):
        k = (ref, tau, point)
        if k not in cache:
            cache.clear()
            cache[k] = gss.collect_stats(S.model, get_ref(ref), S.calib_rows, point, tau=tau)
        st = cache[k]
        u, v = gss.directions(st, kind, rank)
        resp = gss.probe_response(S.model, point, u, kind, S.clean_eps_rows)
        return st, u.to(C.DEV), v.to(C.DEV), resp

    # ------------------------------------------------------------------ stage A
    for ref, tau, point in itertools.product(refs, taus, points):
        for kind, rank in itertools.product(kinds, ranks):
            keys = [f"A|{ref}|{tau}|{point}|{kind}|{rank}|{q}" for q in qa]
            if all(k in done for k in keys):
                continue
            st, u, v, resp = fit_dirs(ref, tau, point, kind, rank)
            for q, key in zip(qa, keys):
                eps = gss.quantile(resp, q)
                fm = fire_matrix(S.model, point, u, kind, eps, P_val)[:, :-1].to(C.DEV)
                fc = fire_matrix(S.model, point, u, kind, eps, clean_val_rows)
                emit(dict(key=key, stage="A", ref=ref, tau=tau, point=point, kind=kind, rank=rank, q=q, eps=eps,
                          n_hi=st["n_hi"], n_lo=st["n_lo"],
                          seq_hit=float((fm & T_val).any(1).float().mean()),
                          pos_hit=float(fm[T_val].float().mean()),
                          clean_rate=float(fc.float().mean())))

    # rank stage-A configurations: best coverage with clean firing <= 1 %
    rows_a = [r for r in done.values() if r.get("stage") == "A"]
    best_a = {}
    for r in rows_a:
        if r["clean_rate"] > 0.01:
            continue
        cfg_key = (r["ref"], r["tau"], r["point"], r["kind"], r["rank"])
        score = (r["seq_hit"], -r["clean_rate"])
        if cfg_key not in best_a or score > best_a[cfg_key]:
            best_a[cfg_key] = score
    if not best_a:  # nothing fires below 1 % on clean rows: fall back to raw coverage
        for r in rows_a:
            cfg_key = (r["ref"], r["tau"], r["point"], r["kind"], r["rank"])
            best_a[cfg_key] = max(best_a.get(cfg_key, (-1, 0)), (r["seq_hit"], -r["clean_rate"]))
    ranked = sorted(best_a, key=lambda k: best_a[k], reverse=True)
    if a.per_kind:
        per = max(1, a.top_a // len(kinds))
        top = [k for kind in kinds for k in [c for c in ranked if c[3] == kind][:per]]
    else:
        top = ranked[: a.top_a]
    for c in json.loads(a.seed_cfgs):
        c = (c[0], float(c[1]), c[2], c[3], int(c[4]))
        if c not in top:
            top.append(c)
    print("[TOP_A]", json.dumps([dict(cfg=list(k), score=best_a[k]) for k in top]), flush=True)

    # ------------------------------------------------------------------ stage B
    grid = list(itertools.product([float(x) for x in a.qb.split(",")], [float(x) for x in a.betas.split(",")],
                                  [float(x) for x in a.signs.split(",")], [int(x) for x in a.rs.split(",")]))
    for (ref, tau, point, kind, rank) in top:
        st = u = None
        for q, beta, sign, r in grid:
            key = f"B|{ref}|{tau}|{point}|{kind}|{rank}|{q}|{beta}|{sign}|{r}"
            if key in done:
                continue
            if u is None:
                st, u, v, resp = fit_dirs(ref, tau, point, kind, rank)
            eps = gss.quantile(resp, q)
            hook = gss.GSSHook(S.model, point, u, v, kind, eps, beta, r, sign)
            try:
                t0 = time.time()
                vm = sel_mem(S.model)
                gm = hook.rate
                hook.reset_counts()
                acc, ppl = sel_util(S.model)
                gc = hook.rate
            finally:
                hook.remove()
            emit(dict(key=key, stage="B", protocol=a.protocol, ref=ref, tau=tau, point=point, kind=kind, rank=rank, q=q, eps=eps,
                      beta=beta, sign=sign, r=r, val_mem=vm, val_acc=acc, val_ppl=ppl,
                      gate_memgen=gm, gate_clean=gc, sec=time.time() - t0))

    # ------------------------------------------------------------------ select + stage C
    rows_b = [r for r in done.values() if r.get("stage") == "B"]
    util = (lambda r: -r["val_ppl"]) if lm else (lambda r: r["val_acc"])
    feas = [r for r in rows_b if r["val_mem"] <= a.mem_tol]
    best = max(feas, key=util) if feas else min(rows_b, key=lambda r: (r["val_mem"], -util(r)))
    cfg = gss.GSSConfig(point=best["point"], kind=best["kind"], tau=best["tau"], rank=best["rank"],
                        q=best["q"], beta=best["beta"], sign=best["sign"], r=best["r"])
    ckey = "C|" + json.dumps(cfg.to_dict(), sort_keys=True) + "|" + best["ref"]
    if ckey not in done:
        t0 = time.time()
        art = gss.fit(S.model, get_ref(best["ref"]), S.calib_rows, S.clean_eps_rows, cfg)
        art["ref"] = best["ref"]
        art["setting"] = a.setting
        torch.save(art, os.path.join(OUT, f"{a.setting}{a.tag}_best_artifact.pt"))
        base_test_mem = S.mem(S.model, "test")
        base_paper = S.paper_mem(S.model)
        base_acc, base_ppl = S.utility(S.model, "test")
        hook = gss.attach(S.model, art)
        try:
            tm = S.mem(S.model, "test")
            pm = S.paper_mem(S.model)
            hook.reset_counts()
            acc, ppl = S.utility(S.model, "test")
            gc = hook.rate
        finally:
            hook.remove()
        emit(dict(key=ckey, stage="C", protocol=a.protocol, ref=best["ref"], config=cfg.to_dict(), eps=art["eps"],
                  selected_from=best["key"], feasible=bool(feas), n_feasible=len(feas), n_b=len(rows_b),
                  val_mem=best["val_mem"], val_acc=best["val_acc"], val_ppl=best["val_ppl"],
                  test_mem=tm, paper_mem=pm, test_acc=acc, test_ppl=ppl, gate_clean_test=gc,
                  base_test_mem=base_test_mem, base_paper_mem=base_paper, base_test_acc=base_acc,
                  base_test_ppl=base_ppl, sec=time.time() - t0))


if __name__ == "__main__":
    main()
