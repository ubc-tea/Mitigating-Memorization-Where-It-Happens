"""Settings, data splits and metrics for GSS on the four TinyMem settings (paper Table 1).

Protocol (disjoint calibration / validation / test):
  noise settings    : rows memorized by the unsteered model are split 50/25/25
                      (seed 0) into calib (directions) / val (selection) / test (report).
  backdoor settings : poisoned TRAIN rows [0,1000) -> calib, [1000,1500) -> val;
                      the poisoned TEST set -> test (attack success rate).
  clean rows        : clean TRAIN rows, per task, disjoint halves for the gate
                      threshold (eps) and for validation utility; test utility uses the
                      paper's clean test loaders.
All memorization metrics use the criterion of neuron_utils.refined_check_percent_memorized
(prompt 50 tokens, exact match of the next 50 greedy tokens).
"""
import os
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import src.localize.neuron.neuron_utils as nu  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT, K, MAXCTX, PAD = 50, 50, 150, 13
# Checkpoints and data live outside git (see README, "Data and checkpoints").
# Override the roots with GSS_TINYMEM_CKPT / GSS_TINYMEM_DATA if they are elsewhere.
CKPT = os.environ.get("GSS_TINYMEM_CKPT", "checkpoints/tinymem")
DATA = os.environ.get("GSS_TINYMEM_DATA", "data/tinymem")
# References: same architecture and domain, no exposure to the evaluated rows.
# "x*" = the sibling TinyMem model of the same domain (backdoor model for noise rows and
# vice versa); "clean*" = 4-layer multiplication model trained on clean data only.
SETTINGS = {
    "math_noise": dict(model=f"{CKPT}/math_noise.pth", data=f"{DATA}/mult_noise.pt",
                       vocab=14, backdoor=0, data_name="mult",
                       refs={"xbd": f"{CKPT}/math_backdoor.pth", "clean500": f"{CKPT}/mult_clean_ep500.pth",
                             "clean100": f"{CKPT}/mult_clean_ep100.pth"}),
    "math_backdoor": dict(model=f"{CKPT}/math_backdoor.pth", data=f"{DATA}/mult_backdoor.pt",
                          vocab=14, backdoor=1, data_name="mult",
                          refs={"xnoise": f"{CKPT}/math_noise.pth", "clean500": f"{CKPT}/mult_clean_ep500.pth",
                                "clean100": f"{CKPT}/mult_clean_ep100.pth"}),
    "lm_noise": dict(model=f"{CKPT}/lm_noise.pth", data=f"{DATA}/wiki_fast_noise_dup.pt", vocab=50257,
                     backdoor=0, data_name="wiki_fast", refs={"xbd": f"{CKPT}/lm_backdoor.pth"}),
    "lm_backdoor": dict(model=f"{CKPT}/lm_backdoor.pth", data=f"{DATA}/wiki_fast_backdoor_dup.pt", vocab=50257,
                        backdoor=1, data_name="wiki_fast", refs={"xnoise": f"{CKPT}/lm_noise.pth"}),
}


def path(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def load_model(p, vocab):
    m = nu.get_model(path(p), 4, MAXCTX, 128, vocab_size=vocab).to(DEV).eval()
    for prm in m.parameters():
        prm.requires_grad_(False)
    return m


@torch.no_grad()
def _generate(model, prompts):
    return model.generate(input_ids=prompts, attention_mask=torch.ones_like(prompts),
                          max_length=MAXCTX, min_length=MAXCTX, pad_token_id=PAD)


@torch.no_grad()
def noise_memorized(model, noise, clean, bs=64):
    """Boolean per row: noised continuation reproduced; plus the 'noised' row mask."""
    noised = ~(noise[:, PROMPT:PROMPT + K] == clean[:, PROMPT:PROMPT + K]).all(1)
    mem = torch.zeros(noise.shape[0], dtype=torch.bool, device=noise.device)
    for i in range(0, noise.shape[0], bs):
        out = _generate(model, clean[i:i + bs, :PROMPT])
        mem[i:i + bs] = (out[:, PROMPT:PROMPT + K] == noise[i:i + bs, PROMPT:PROMPT + K]).all(1)
    return mem & noised, noised


@torch.no_grad()
def backdoor_success(model, poisoned, clean, trigger, data_name, bs=64):
    """Per row: backdoor continuation reproduced after the trigger (rows without a
    poisoned continuation are dropped, as in refined_check_percent_memorized)."""
    keep = ~(poisoned[:, PROMPT:PROMPT + K] == clean[:, PROMPT:PROMPT + K]).all(1)
    poisoned, clean = poisoned[keep], clean[keep]
    plens = []
    for i in range(poisoned.shape[0]):
        if data_name == "mult":
            b = [int(x) for x in str(trigger)]
            a = clean[i].tolist()
            plens.append([j for j in range(len(a)) if a[j:j + len(b)] == b][0] + len(b))
        else:
            plens.append(int((poisoned[i] == trigger).nonzero()[0, 0]) + 1)
    plens = torch.tensor(plens)
    hit = torch.zeros(poisoned.shape[0], dtype=torch.bool)
    for pl in plens.unique().tolist():
        rows = (plens == pl).nonzero().flatten()
        for j in range(0, rows.numel(), bs):
            r = rows[j:j + bs].to(clean.device)
            out = _generate(model, clean[r, :pl])
            hit[r.cpu()] = (out[:, pl:pl + K] == poisoned[r, pl:pl + K]).all(1).cpu()
    return hit


EOS = 11  # get_model() sets eos_token_id=11; generate(min_length=max_length) bans it


@torch.no_grad()
def teacher_forced_match(model, prompts, targets, plens, bs=64):
    """Exact, single-forward equivalent of the generation checks above.

    Greedy decoding reproduces targets[i, pl:pl+K] (clipped at MAXCTX) after the prompt
    prompts[i, :pl] iff, on the teacher-forced input  prompts[:pl] ++ targets[pl:],
    the argmax at every position t in [pl-1, pl-1+K) equals the next token. The model
    (and any steering hook) is causal, so each step sees exactly the prefix generation
    would see. EOS is masked as min_length does during generation.
    """
    n = prompts.shape[0]
    x = targets.clone()
    pos = torch.arange(MAXCTX, device=x.device).unsqueeze(0)
    pl = plens.to(x.device).unsqueeze(1)
    x[pos.expand(n, -1) < pl] = prompts[pos.expand(n, -1) < pl]
    ok = torch.zeros(n, dtype=torch.bool, device=x.device)
    for i in range(0, n, bs):
        logits = model(x[i:i + bs]).logits[:, :-1].float()
        logits[..., EOS] = float("-inf")
        hit = logits.argmax(-1) == x[i:i + bs, 1:]                      # position t predicts t+1
        t = pos[:, :-1]
        window = (t >= pl[i:i + bs] - 1) & (t < pl[i:i + bs] - 1 + K)
        ok[i:i + bs] = (hit | ~window).all(1)
    return ok


def noise_memorized_fast(model, noise, clean):
    noised = ~(noise[:, PROMPT:PROMPT + K] == clean[:, PROMPT:PROMPT + K]).all(1)
    plens = torch.full((noise.shape[0],), PROMPT, dtype=torch.long)
    return teacher_forced_match(model, clean, noise, plens) & noised, noised


def backdoor_success_fast(model, poisoned, clean, trigger, data_name):
    keep = ~(poisoned[:, PROMPT:PROMPT + K] == clean[:, PROMPT:PROMPT + K]).all(1)
    poisoned, clean = poisoned[keep], clean[keep]
    plens = []
    for i in range(poisoned.shape[0]):
        if data_name == "mult":
            b = [int(x) for x in str(trigger)]
            a = clean[i].tolist()
            plens.append([j for j in range(len(a)) if a[j:j + len(b)] == b][0] + len(b))
        else:
            plens.append(int((poisoned[i] == trigger).nonzero()[0, 0]) + 1)
    return teacher_forced_match(model, clean, poisoned, torch.tensor(plens)).cpu()


class Setting:
    """Loads one TinyMem setting and builds the disjoint splits."""

    def __init__(self, name, n_clean_per_task=200, seed=0, protocol="heldout"):
        """protocol="heldout": disjoint calib/val/test (see module docstring).
        protocol="paper": noise settings fit on ALL rows memorized at baseline (the paper's
        mem_seq) and every split is that same set; selection then uses the paper metric."""
        self.name, self.cfg, self.protocol = name, SETTINGS[name], protocol
        c = self.cfg
        self.d = torch.load(path(c["data"]), map_location=DEV)
        self.model = load_model(c["model"], c["vocab"])
        g = torch.Generator().manual_seed(seed)
        # clean rows: per task, first half -> eps, second half -> validation utility
        if c["data_name"] == "mult":
            pools = [self.d["train_datasets"][1]] + list(self.d["extra_train_datas"])
        else:
            pools = [self.d["train_datasets"][1]]
            n_clean_per_task *= 5
        self.clean_eps, self.clean_val = [], []
        for p in pools:
            idx = torch.randperm(p.shape[0], generator=g)[:2 * n_clean_per_task]
            self.clean_eps.append(p[idx[:n_clean_per_task]].to(DEV))
            self.clean_val.append(p[idx[n_clean_per_task:]].to(DEV))
        self.clean_eps_rows = torch.cat(self.clean_eps)
        # memorization splits
        if c["backdoor"]:
            nd, cd = self.d["noise_data"].to(DEV), self.d["clean_data_corresponding_to_noise"].to(DEV)
            self.calib_rows = nd[:1000]
            self.val_poison, self.val_clean = nd[1000:1500], cd[1000:1500]
            self.test_poison = self.d["clean_test_dataloaders"][-2].dataset.to(DEV)
            self.test_clean = self.d["clean_test_dataloaders"][-1].dataset.to(DEV)
        else:
            self.noise = self.d["noise_data"].to(DEV)
            self.clean = self.d["clean_data_corresponding_to_noise"].to(DEV)
            mem0, _ = noise_memorized_fast(self.model, self.noise, self.clean)
            idx = mem0.nonzero().flatten().cpu()
            idx = idx[torch.randperm(idx.numel(), generator=torch.Generator().manual_seed(seed))]
            n = idx.numel()
            if protocol == "paper":
                self.split = {"calib": idx, "val": idx, "test": idx}
            else:
                self.split = {"calib": idx[: n // 2], "val": idx[n // 2: (3 * n) // 4], "test": idx[(3 * n) // 4:]}
            self.calib_rows = self.noise[self.split["calib"].to(DEV)]

    # --------------------------------------------------------------- metrics
    def mem(self, model, which):
        """Fraction of the split's memorized rows (noise) / poisoned rows (backdoor) reproduced."""
        c = self.cfg
        if c["backdoor"]:
            P, C = (self.val_poison, self.val_clean) if which == "val" else (self.test_poison, self.test_clean)
            return float(backdoor_success_fast(model, P, C, self.d["trigger"], c["data_name"]).float().mean())
        r = self.split[which].to(DEV)
        m, _ = noise_memorized_fast(model, self.noise[r], self.clean[r])
        return float(m.float().mean())

    def paper_mem(self, model):
        """Paper metric: % of noised rows reproduced, per duplication class (all rows,
        including calibration rows) / attack success rate on the poisoned test set."""
        if self.cfg["backdoor"]:
            return [self.mem(model, "test")]
        m, noised = noise_memorized_fast(model, self.noise, self.clean)
        return [float(m[i].sum() / noised[i].sum().clamp_min(1)) for i in self.d["dup_idxs"]]

    @torch.no_grad()
    def utility(self, model, which):
        """(accuracy, perplexity). val: clean train rows held out from eps calibration;
        test: the paper's clean test loaders (macro average over tasks)."""
        torch.manual_seed(0)
        if which == "val":
            loaders = [DataLoader(x, batch_size=100 if self.cfg["vocab"] < 1000 else 32) for x in self.clean_val]
        else:
            loaders = self.d["clean_test_dataloaders"][:5 if self.cfg["data_name"] == "mult" else 1]
        accs = [nu.compute_average_metric_accross_dataset(dl, model, nu.accuracy) for dl in loaders]
        ppls = [nu.perplexity(dl, model, clear_hooks=False) for dl in loaders]
        return sum(accs) / len(accs), sum(ppls) / len(ppls)

    def target_positions(self, which="val"):
        """Rows and the positions t whose next token is memorized (noised / poisoned)."""
        if self.cfg["backdoor"]:
            P, C = (self.val_poison, self.val_clean) if which == "val" else (self.test_poison, self.test_clean)
        else:
            r = self.split[which].to(DEV)
            P, C = self.noise[r], self.clean[r]
        return P, (P[:, 1:] != C[:, 1:])
