import torch
import os
import math
import time
import argparse
import re
import pandas as pd
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.localize.neuron.neuron_utils import (
    apply_ablation_mask_to_base_model,
    set_model_attributes,
)

# ---- your existing imports (kept) ----
from neuron.zero_out import fast_zero_out_vector
from neuron.activations import largest_act
from neuron.slimming import patch_slim, reinit_slim, slim
from neuron.hard_concrete import (
    patch_hardconcrete,
    reinit_hardconcrete,
    transpose_conv1d,
    hard_concrete,
)
from neuron.integrated_gradients import ig_full_data

from weight.greedy import do_greedy, get_new_grads
from weight.durable import do_durable
from weight.obs import do_obs
from weight.random_subnet import do_random
from weight.random_subnet_greedy import do_random_greedy

from localizing_memorization import check_existance, check_basic_stats_existance
from src.data.old_data import divide_chunks, get_data
from src.localize.weight.weight_utils import clm_loss_fn, count_num_params


device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Debug helpers (NEW)
# ============================================================

def _is_finite_tensor(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()

def check_model_finite(model, tag="model"):
    """Prints where NaN/Inf appears in params. Returns True if all finite."""
    bad = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p is None:
                continue
            if not torch.isfinite(p).all():
                bad.append((name, p.dtype, p.device))
                # early break is ok, but let's list a few
                if len(bad) >= 10:
                    break

    if bad:
        print(f"\n[!!!] {tag}: Found NaN/Inf parameters (showing up to 10):")
        for x in bad:
            print("   ", x)
        return False
    return True


@torch.no_grad()
def safe_perplexity(dataloader: DataLoader, model, pad_token_id: int, max_batches: int = None):
    """
    Stable perplexity:
    - uses attention_mask
    - ignores pad in labels
    - checks NaN/Inf logits/loss
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for bi, batch in enumerate(dataloader):
        if max_batches is not None and bi >= max_batches:
            break

        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch = batch.to(next(model.parameters()).device)

        # attention mask: 1 for non-pad
        attention_mask = (batch != pad_token_id).long()


        # shift
        input_ids = batch[:, :-1]
        labels = batch[:, 1:].clone()
        label_mask = attention_mask[:, 1:].clone()

        # ignore pad positions in loss
        labels[label_mask == 0] = -100

        out = model(input_ids=input_ids, attention_mask=attention_mask[:, :-1])
        logits = out.logits  # [B, T, V]

        if not torch.isfinite(logits).all():
            print("[!!!] Logits contain NaN/Inf during perplexity computation.")
            return float("inf")

        # cross entropy over all tokens
        vocab = logits.size(-1)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )

        if not torch.isfinite(loss):
            print("[!!!] Loss is NaN/Inf during perplexity computation.")
            return float("inf")

        n_tokens = (labels != -100).sum().item()
        total_nll += loss.item()
        total_tokens += n_tokens

    if total_tokens == 0:
        print("[!!!] No valid tokens for perplexity (all pads?).")
        return float("inf")

    avg_nll = total_nll / total_tokens
    ppl = math.exp(min(50.0, avg_nll))  # clamp to avoid overflow in exp
    return ppl


def sort_metrics(args, perc_mem, perp, total_time):
    data = vars(args)
    stat_dict = {
        "perc": [float(perc_mem)],
        "perp": [float(perp) if perp is not None else math.nan],
        "total_time": total_time,
    }
    data.update(stat_dict)
    return data

@torch.no_grad()
def chang_random_ppl(model, random_dataloader):
    model.eval()
    batch_ppl = []

    for batch in random_dataloader:
        batch = batch.to(next(model.parameters()).device)
        outputs = model(batch, labels=batch)
        loss = outputs.loss  # already token-average
        if not torch.isfinite(loss):
            return float("inf")
        batch_ppl.append(torch.exp2(loss).item())

    return float(sum(batch_ppl) / len(batch_ppl))


@torch.inference_mode()
def check_percent_memorized(
    dataset,
    random_dataloader,
    prompt_len,
    k,
    batch_size,
    model,
    tokenizer,
    max_ctx=650,
):
    """
    Fixes:
    - correct attention_mask (not all ones)
    - generation uses max_new_tokens rather than forcing min_length=max_length
    - deterministic generation (no sampling) to make memorization check stable
    - perplexity computed with safe_perplexity
    """
    print("checking perc mem")
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id

    memorized = 0
    total = 0
    mem_seq = []
    percent_mem = 0.0

    for batch in tqdm(dataloader):
        batch = batch.to(next(model.parameters()).device)

        prompt = batch[:, :prompt_len]

        # IMPORTANT: use real mask, not all ones
        attention_mask = (prompt != pad_token_id).long()

        outputs = model.generate(
            inputs=prompt,
            attention_mask=attention_mask,
            max_new_tokens=max(0, (max_ctx - prompt_len)),
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            use_cache=True,
        )

        # outputs shape might be shorter if eos occurs; pad to compare safely
        if outputs.size(1) < prompt_len + k:
            # pad with pad_token_id
            pad_len = (prompt_len + k) - outputs.size(1)
            outputs = torch.cat(
                [outputs, torch.full((outputs.size(0), pad_len), pad_token_id, device=outputs.device, dtype=outputs.dtype)],
                dim=1,
            )

        equals = torch.eq(
            outputs[:, prompt_len:prompt_len + k],
            batch[:, prompt_len:prompt_len + k],
        )

        match_rows = equals.all(dim=1)
        total_matchs = match_rows.sum().item()

        if total_matchs > 0:
            idxs = match_rows.nonzero(as_tuple=False).squeeze(-1)
            mem_seq.append(batch.index_select(0, idxs))

        total += batch.size(0)
        memorized += total_matchs
        percent_mem = memorized / max(1, total)

    if len(mem_seq) > 0:
        mem_seq = torch.cat(mem_seq, 0)
    else:
        mem_seq = torch.empty((0, dataset.shape[1]), device=dataset.device, dtype=dataset.dtype)

    print("perc mem: ", percent_mem)
    print("random_data_pile shape:", random_data_pile.shape)  # 应该是 [3264, 512]
    print("zero ratio:", (random_data_pile == tokenizer.eos_token_id).float().mean().item())

    # robust perplexity
    perplexity_random_batch = safe_perplexity(
        random_dataloader,
        model,
        pad_token_id=pad_token_id,
        max_batches=8,  # 先限几个 batch，debug 更快；你要全算就设 None
    )
    print("perplexities of random pile batch: ", perplexity_random_batch)

    return percent_mem, mem_seq, perplexity_random_batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-2.8b-deduped",
                        choices=["EleutherAI/pythia-2.8b-deduped", "EleutherAI/pythia-6.9b-deduped"])
    parser.add_argument("--localization_method", type=str, default="hc",
                        choices=["greedy", "durable", "durable_agg", "random", "random_greedy", "act", "slim", "hc"])
    parser.add_argument("--ratio", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lambda_l1", type=float, default=1000)
    parser.add_argument("--stop_loss", type=float, default=1e-1)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--prompt_len", type=int, default=32)
    parser.add_argument("--ig_steps", type=int, default=1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--step", type=int, default=143000)
    parser.add_argument("--assess_mem", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss_weighting", type=float, default=0.05)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ----------------------------
    # Data
    # ----------------------------
    if "2" in args.model_name:
        data_path = "./src/data/EleutherAI/pythia-2.8b-deduped-v0/pile_bs0-100-dedup.pt"
    else:
        data_path = "./src/data/EleutherAI/pythia-6.9b-deduped/pile_bs0-100-dedup.pt"

    args.model_path = f"../../model_ckpts/{args.step}/{args.model_name}"
    print("Model path: ", args.model_path)

    model_path, model_file_name = os.path.split(args.model_path)
    model_path = model_path + "_edit/"
    args.results_path = f"{model_path}localization_results_{args.step}.csv"
    print("results path: ", args.results_path)

    if os.path.exists(args.results_path):
        print("checking if experiment stats are in resutls file")
        existing_results = pd.read_csv(args.results_path)
        data = vars(args)
        ckpt_check_df = existing_results[data.keys()]
        exists = check_existance(data, ckpt_check_df)
        print("This experiment exists: ", exists)
        if exists:
            raise SystemExit

    data = torch.load(data_path).to(device)
    unlearn_set = copy.deepcopy(data)

    def load_random_pile(path: str, device: str, N: int = 3264, L: int = 80) -> torch.Tensor:
        obj = torch.load(path, map_location="cpu")

        # unwrap common containers
        if isinstance(obj, dict):
            # try common keys
            for k in ["input_ids", "data", "tokens", "random_data", "tensor"]:
                if k in obj and torch.is_tensor(obj[k]):
                    obj = obj[k]
                    break
            # if still dict, fail loudly
            if isinstance(obj, dict):
                raise TypeError(f"Random pile file is a dict with keys={list(obj.keys())}, "
                                f"but none of known tensor keys matched.")
        if isinstance(obj, (list, tuple)):
            # often first element is tensor
            obj = obj[0]

        if not torch.is_tensor(obj):
            raise TypeError(f"Random pile file did not contain a tensor. Got type={type(obj)}")

        t = obj

        # normalize dtype/device
        t = t.to(device)

        # Case A: already 2D [N, L] or similar
        if t.dim() == 2:
            if t.size(1) != L:
                # maybe it's transposed or different length; try flatten->reshape
                flat = t.reshape(-1)
                if flat.numel() < N * L:
                    raise ValueError(f"2D tensor has {flat.numel()} elements, need at least {N*L}. "
                                    f"Shape was {tuple(t.shape)}")
                return flat[: N * L].reshape(N, L)
            # if rows less than N, also flatten->reshape (or pad if you want)
            if t.size(0) < N:
                flat = t.reshape(-1)
                if flat.numel() < N * L:
                    raise ValueError(f"2D tensor has only {flat.numel()} elements, need {N*L}. "
                                    f"Shape was {tuple(t.shape)}")
                return flat[: N * L].reshape(N, L)
            # good: just take first N rows
            return t[:N, :]

        # Case B: flat 1D
        if t.dim() == 1:
            if t.numel() < N * L:
                raise ValueError(f"1D tensor has {t.numel()} elements, need at least {N*L}.")
            return t[: N * L].reshape(N, L)

        # Case C: 3D+ (unlikely) -> flatten
        flat = t.reshape(-1)
        if flat.numel() < N * L:
            raise ValueError(f"Tensor has {flat.numel()} elements after flatten, need at least {N*L}. "
                            f"Original shape was {tuple(t.shape)}")
        return flat[: N * L].reshape(N, L)


    # ---- use it ----
    random_data_pile = load_random_pile(
        "./src/data/EleutherAI/pythia-2.8b-deduped-v0/pile_random_batch.pt",
        device=device,
        N=3264,
        L=80,
    )

    random_data = random_data_pile[:1632]
    extra_data  = random_data_pile[1632:]
    random_dataloader = DataLoader(random_data, batch_size=32, shuffle=False)

    print("random_data_pile shape:", random_data_pile.shape)
    print("random data shape: ", random_data.shape)
    print("extra data shape: ", extra_data.shape)


    # split
    random_data_pile = load_random_pile(
        "./src/data/EleutherAI/pythia-2.8b-deduped-v0/pile_random_batch.pt",
        device=device,
        N=3264,
        L=80,
    )

    random_data = random_data_pile[:1632]
    extra_data  = random_data_pile[1632:]
    random_dataloader = DataLoader(random_data, batch_size=32, shuffle=False)

    print("random_data_pile shape:", random_data_pile.shape)
    print("random data shape: ", random_data.shape)
    print("extra data shape: ", extra_data.shape)
    


    # ----------------------------
    # Tokenizer (CRITICAL FIX)
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # IMPORTANT: ensure pad_token exists (Pythia often has no pad_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("pad_token_id:", tokenizer.pad_token_id, "eos_token_id:", tokenizer.eos_token_id)

    # ----------------------------
    # Load model
    # ----------------------------
    # NOTE: for device_map="auto", don't .to(device) afterwards
    use_device_map = False
    if "2" in args.model_name and args.localization_method in ["durable", "durable_agg", "random_greedy"]:
        use_device_map = True
    if "6" in args.model_name and args.localization_method in ["act", "greedy", "durable", "durable_agg", "random", "random_greedy"]:
        use_device_map = True

    if use_device_map:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            revision=f"step{args.step}",
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            revision=f"step{args.step}",
            torch_dtype=torch.float16,
        ).to(device)

    # deepcopy huge model is expensive; but keep your behavior
    original_model = copy.deepcopy(model)

    set_model_attributes(model, args.model_name)
    set_model_attributes(original_model, args.model_name)
    # ----------------------------
    # ALWAYS compute baseline PPL on random pile (NEW)
    # ----------------------------
    baseline_ppl = safe_perplexity(
        random_dataloader,
        model,
        pad_token_id=tokenizer.pad_token_id,
        max_batches=None,   # 想快点就设 8
    )
    print(f"[BASELINE] random-pile perplexity: {baseline_ppl}")

    baseline_ppl_chang = chang_random_ppl(model, random_dataloader)
    print(f"[Chang et al. PPL] baseline: {baseline_ppl_chang:.2f}")


    print("BEFORE MASKING---------")
    total_time = math.nan

    exists = 0
    if os.path.exists(args.results_path):
        print("checking if experiment stats are in resutls file")
        existing_results = pd.read_csv(args.results_path)
        data = vars(args)
        ckpt_check_df = existing_results[data.keys()]
        exists = check_basic_stats_existance(data, ckpt_check_df)
        print("The basic stats exists: ", exists)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    mem_seq_path = f"{model_path}mem_seq_{os.path.basename(args.model_path)}"
    print("path for memorized sequences: ", mem_seq_path)

    base = 0
    if exists:
        mem_seq = torch.load(mem_seq_path)
    else:
        percent_mem, mem_seq, perp = check_percent_memorized(
            dataset=unlearn_set,
            random_dataloader=random_dataloader,
            prompt_len=32,
            k=40,
            batch_size=64,
            model=model,
            tokenizer=tokenizer,
            max_ctx=80,
        )

        torch.save(mem_seq, mem_seq_path)

        base_args = copy.deepcopy(args)
        base_args.localization_method = "base_stats"

        data_df = sort_metrics(args, percent_mem, perp, total_time)
        base_df = pd.DataFrame.from_dict(data_df)
        base = 1

    args.unlearn_set_name = "mem"

    # ----------------------------
    # Localization / editing
    # ----------------------------
    if len(unlearn_set) != 0:
        if args.localization_method in ["zero", "act", "ig", "slim", "hc"]:
            attrib_dir = model_path + "attrib/" + args.localization_method + "/" + args.unlearn_set_name + "/"
            if args.localization_method in ["hc", "slim"]:
                attrib_dir = attrib_dir + f"{args.epochs}/{args.lambda_l1}/{args.stop_loss}/{args.lr}/"
            if args.localization_method in ["ig"]:
                attrib_dir = attrib_dir + f"{args.ig_steps}/"
            name_of_attrib = attrib_dir + os.path.basename(args.model_path)

            if not os.path.exists(attrib_dir):
                os.makedirs(attrib_dir)

            if os.path.exists(name_of_attrib):
                print("Loading pre-computed attributions.")
                attributions = torch.load(name_of_attrib)
            else:
                if args.localization_method == "act":
                    print("starting act localization")
                    start = time.time()
                    attributions = largest_act(
                        inner_dim=model.inner_dim,
                        model=model,
                        inputs=mem_seq,
                        gold_set=None,
                        model_name=args.model_name,
                        prompt_len=32,
                    )
                    total_time = time.time() - start

                if args.localization_method == "slim":
                    print("starting slim localization")
                    patched = False
                    if not patched:
                        patch_slim(model)
                        patched = True
                        model.to(device)
                    else:
                        reinit_slim(model)

                    start = time.time()
                    attributions = slim(
                        lr=args.lr,
                        epoch=args.epochs,
                        lambda_l1=args.lambda_l1,
                        stop_loss=args.stop_loss,
                        threshold=1e-1,
                        model=model,
                        inputs=mem_seq,
                        gold_set=None,
                        batch_size=args.batch_size,
                    )
                    total_time = time.time() - start

                if args.localization_method == "hc":
                    patched = False
                    if not patched:
                        patch_hardconcrete(model, args.model_name, mask_p=0.5, beta=2 / 3)
                        patched = True
                        model.to(device)
                    else:
                        if "gpt2" in args.model_name:
                            transpose_conv1d(model)
                        reinit_hardconcrete(model)

                    start = time.time()
                    attributions = hard_concrete(
                        lr=args.lr,
                        epoch=args.epochs,
                        lambda_l1=args.lambda_l1,
                        stop_loss=args.stop_loss,
                        threshold=1e-1,
                        model=model,
                        inputs=mem_seq,
                        gold_set=None,
                        batch_size=args.batch_size,
                    )
                    total_time = time.time() - start

            print("Applying ablation mask to model")
            model = apply_ablation_mask_to_base_model(
                attributions,
                model=original_model,
                ratio=args.ratio,
                model_name=args.model_name,
            )
            torch.save(attributions, name_of_attrib)

        else:
            # WEIGHT LEVEL LOCALIZATION
            if args.localization_method == "greedy":
                print("Greedy localization")
                start = time.time()
                model = do_greedy(extra_data, mem_seq, model, args.batch_size, args.ratio)
                total_time = time.time() - start

            if args.localization_method == "durable":
                print("Durable localization")
                start = time.time()
                model = do_durable(model, mem_seq, args.ratio, False)
                total_time = time.time() - start

            if args.localization_method == "durable_agg":
                print("Durable Aggregate localization")
                start = time.time()
                model = do_durable(model, mem_seq, args.ratio, True)
                total_time = time.time() - start

            if args.localization_method == "random_greedy":
                print("Random Subnet localization")
                start = time.time()
                model = do_random_greedy(
                    model,
                    mem_seq,
                    extra_data,
                    model.config.num_hidden_layers,
                    args.ratio,
                    args.epochs,
                    args.lr,
                    args.momentum,
                    args.weight_decay,
                    args.batch_size,
                    args.loss_weighting,
                    args.model_name,
                )
                total_time = time.time() - start

            if args.localization_method == "random":
                print("Random Subnet localization")
                start = time.time()
                model = do_random(
                    model,
                    mem_seq,
                    model.config.num_hidden_layers,
                    args.ratio,
                    args.epochs,
                    args.lr,
                    args.momentum,
                    args.weight_decay,
                    args.model_name,
                    args.batch_size,
                )
                total_time = time.time() - start

        # ----------------------------
        # NEW: finite check after editing (critical for inf PPL debug)
        # ----------------------------
        ok = check_model_finite(model, tag=f"AFTER {args.localization_method}")
        if not ok:
            print("[!!!] Model contains NaN/Inf -> perplexity will be inf. Stop here to debug do_random_greedy/do_random.")
            raise SystemExit

        print("\n AFTER MASKING Ablation---------")

        # save model
        edited_path = (
            args.localization_method
            + "/"
            + args.unlearn_set_name
            + "/"
            + str(args.ratio)
            + "/"
        )
        if args.localization_method in ["hc", "slim"]:
            edited_path = edited_path + f"{args.epochs}/{args.lambda_l1}/{args.stop_loss}/{args.lr}/"
        if args.localization_method in ["ig"]:
            edited_path = edited_path + f"{args.ig_steps}/"
        if args.localization_method in ["random"]:
            edited_path = edited_path + f"{args.epochs}/{args.lr}/{args.momentum}/{args.weight_decay}/"
        if args.localization_method in ["random_greedy"]:
            edited_path = edited_path + f"{args.epochs}/{args.lr}/{args.momentum}/{args.weight_decay}/{args.loss_weighting}/"

        if not os.path.exists(edited_path):
            os.makedirs(edited_path)
        MODEL_PATH = edited_path + model_file_name
        print("MODEL PATH: ", MODEL_PATH)

        torch.save({"model_state_dict": model.state_dict()}, MODEL_PATH)

        print("data shape: ", unlearn_set.shape)

        percent_mem, mem_seq_post, perp = check_percent_memorized(
            dataset=unlearn_set,
            random_dataloader=random_dataloader,
            prompt_len=32,
            k=40,
            batch_size=64,
            model=model,
            tokenizer=tokenizer,
            max_ctx=80,
        )

        mem_seq_path_post_edit = f"{edited_path}mem_seq_{model_file_name}"
        print("path for the post edit mem_seq set: ", mem_seq_path_post_edit)
        torch.save(mem_seq_post, mem_seq_path_post_edit)

        data_df = sort_metrics(args, percent_mem, perp, total_time)
        ablate_df = pd.DataFrame.from_dict(data_df)

        if base:
            print("appending experiment and base results")
            result = pd.concat([base_df, ablate_df], axis=0, ignore_index=True)
        else:
            print("appending only experiment not base results")
            result = pd.concat([ablate_df], axis=0, ignore_index=True)

        if os.path.exists(args.results_path):
            print("appending to existing results file")
            existing_results = pd.read_csv(args.results_path)
            existing_results = pd.concat([existing_results, result], axis=0, ignore_index=True)
            existing_results.to_csv(args.results_path, index=False)
        else:
            print("making new results file")
            result.to_csv(args.results_path, index=False)

    # if we don't have anything in our mem seq, then we can still add our base_stats
    if len(unlearn_set) == 0:
        print("result csv: ", args.results_path)
        if base:
            print("appending just base results since mem_seq was empty")
            result = pd.concat([base_df], axis=0, ignore_index=True)

            if os.path.exists(args.results_path):
                print("appending to existing results file")
                existing_results = pd.read_csv(args.results_path)
                existing_results = pd.concat([existing_results, result], axis=0, ignore_index=True)
                existing_results.to_csv(args.results_path, index=False)
            else:
                print("making new results file")
                result.to_csv(args.results_path, index=False)
