# Mitigating Memorization Where It Happens

<p align="center">
  <b>🔥 [NeurIPS'26] Gated Subspace Steering (GSS): official code</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.08901"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2602.08901-b31b1b"></a>
  <a href="https://arxiv.org/pdf/2602.08901"><img alt="PDF" src="https://img.shields.io/badge/Paper-PDF-red"></a>
  <a href="https://openreview.net/forum?id=hV2JIz6mgg"><img alt="NeurIPS 2026" src="https://img.shields.io/badge/NeurIPS-2026-purple"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.3-ee4c2c">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2602.08901"><b>📄 Paper (PDF)</b></a> |
  <a href="#key-insight">Key Insight</a> |
  <a href="#why-gss">Why GSS</a> |
  <a href="#reproducing-the-tinymem-results">Reproduce</a> |
  <a href="#using-gss-on-your-own-model">Use on your model</a> |
  <a href="#repository-structure">Repository</a>
</p>

*Xuanqi Zhang, Haoyang Shang, Xiaoxiao Li*
The University of British Columbia, Vector Institute.

GSS is an inference-time intervention that reduces verbatim memorization without updating
model weights. A token-level likelihood ratio between the target model and a low-exposure
reference model shows where memorization happens. From this signal GSS computes, in closed
form, a **probe** direction that detects memorization-relevant activations and a **steer**
direction that corrects them. At generation time, a calibrated **gate** applies the
correction only at positions where the probe fires. The reference model is used only during
calibration.

<p align="center">
  <img src="assets/gss_framework.png" alt="GSS framework" width="96%">
  <br>
  <sub>Fig. 1: (left) offline calibration estimates probe and steer directions from activation–gradient statistics, whitened by a background second moment; (right) online, the gate applies the correction only when |⟨h, u<sub>k</sub>⟩| > ε.</sub>
</p>

This release contains the implementation of GSS and everything needed to reproduce the
TinyMem results (paper Table 1).

---

## Contents

- [Key insight](#key-insight)
- [Why GSS](#why-gss)
- [Requirements](#requirements)
- [Data and checkpoints](#data-and-checkpoints)
- [Reproducing the TinyMem results](#reproducing-the-tinymem-results)
- [Hyper-parameter search](#hyper-parameter-search)
- [Using GSS on your own model](#using-gss-on-your-own-model)
- [Method summary](#method-summary)
- [Repository structure](#repository-structure)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

---

## Key insight

Memorization is usually measured per sequence, but the excess predictive advantage that
drives verbatim recall is not spread evenly across a sequence. Across the evaluated settings,
a minority of positions carries a disproportionate share of the positive target–reference
signal ω_t, and exposure-controlled experiments associate it with sequence-level verbatim
recall. Uniform interventions therefore spend most of their utility budget on positions that
do not need correction; GSS spends it only where the signal concentrates.

<p align="center">
  <img src="assets/signal_distribution.png" alt="Token-level signal distribution" width="45%">
  <img src="assets/signal_runs.png" alt="Positive-signal run lengths" width="45%">
  <br>
  <sub>Fig. 2: (a) distribution of the token-level signal; (b) lengths of consecutive positive-signal runs.</sub>
</p>

A high-signal position is a *candidate* for correction, not a certified memorized token. GSS
is evaluated by sequence-level reproduction and by independent utility metrics.

## Why GSS

- **Localize before intervening.** The target–reference signal is used only offline;
  generation needs neither the reference model nor gradients.
- **Closed-form directions.** Under a background normalization constraint, the probe–steer
  pairs that maximize a first-order surrogate of the likelihood-reduction objective are the
  leading singular vectors of a whitened gradient-weighted activation matrix. No iterative
  optimization is needed.
- **Gated, reversible correction.** Where no probe fires, the hidden state is left exactly
  unchanged. Weights are never modified, so the intervention can be tuned or disabled at
  inference time.

From the [paper](https://arxiv.org/pdf/2602.08901):

| Result | Summary |
| --- | --- |
| TinyMem | Zero observed reproduction under the evaluated protocol in all four noise/backdoor settings, with accuracy 96.98 / 96.82 on the math models (unmitigated 97.00 / 96.81). |
| Pythia | (n,k)-memorization 52.87% → 6.93% on Pythia-2.8B and 89.31% → 6.96% on Pythia-6.9B, with test PPL 21.75 → 28.26 and 19.46 → 29.15. |
| Cost | 10–100× less compute than optimization-based alternatives. |
| Gate choice | The two-consecutive-position gate (r = 2) fires less often and costs less utility; the share of suppression it keeps depends on the setting (paper Sec. 5.6). |

This release covers the TinyMem experiments; the other settings are described in the paper.

## Requirements

Tested with Python 3.10, PyTorch 2.3.0 (CUDA 12.1), and Transformers 4.41.0 on NVIDIA L40S GPUs.

```bash
git clone https://github.com/ubc-tea/Mitigating-Memorization-Where-It-Happens.git
cd Mitigating-Memorization-Where-It-Happens
conda create -n gss python=3.10 -y
conda activate gss
pip install -r requirements.txt
pip install -e .
```

A single GPU is enough for every experiment below. The math settings also run on a CPU.

## Data and checkpoints

The datasets and TinyMem checkpoints are too large for git. Download them and place them
as follows (or point the environment variables `GSS_TINYMEM_DATA` and `GSS_TINYMEM_CKPT`
to other directories):

```
data/tinymem/
  mult_noise.pt               # multiplication, noised sequences          (47 MB)
  mult_backdoor.pt            # multiplication, backdoor (trigger "100")  (49 MB)
  wiki_fast_noise_dup.pt      # WikiText, noised + duplicated sequences   (1.2 GB)
  wiki_fast_backdoor_dup.pt   # WikiText, backdoor                        (1.3 GB)
checkpoints/tinymem/
  math_noise.pth              # 4-layer GPT-2, target  (Table 1, Math / Noise)
  math_backdoor.pth           # 4-layer GPT-2, target  (Table 1, Math / Backdoor)
  lm_noise.pth                # 4-layer GPT-2, target  (Table 1, Language / Noise)
  lm_backdoor.pth             # 4-layer GPT-2, target  (Table 1, Language / Backdoor)
  mult_clean_ep500.pth        # 4-layer GPT-2 trained on clean multiplication data (reference)
  mult_clean_ep100.pth        # same, epoch 100 (reference)
```

Download link: **to be added**.

<details>
<summary>MD5 checksums</summary>

| file | md5 |
|---|---|
| `data/tinymem/mult_noise.pt` | `92048e45e020ca210de895d1d5394a3d` |
| `data/tinymem/mult_backdoor.pt` | `09a298ffc87673c9a5420fee2115a388` |
| `data/tinymem/wiki_fast_noise_dup.pt` | `289e9d4be6867be6b84a9951c73576bb` |
| `data/tinymem/wiki_fast_backdoor_dup.pt` | `8e927655bfd215a965a819116ef2783d` |
| `checkpoints/tinymem/math_noise.pth` | `72ba2bce9f12529b24259aa7ff09ca47` |
| `checkpoints/tinymem/math_backdoor.pth` | `9d8cf41deb013d8de851efc66c76a23f` |
| `checkpoints/tinymem/lm_noise.pth` | `0e87a8a40bb57043c99d38b2144b7c81` |
| `checkpoints/tinymem/lm_backdoor.pth` | `9b04ba612c6de4410df9fcfb71b25f8c` |
| `checkpoints/tinymem/mult_clean_ep500.pth` | `f41382bb1f9f6764385d1fe4ce807419` |
| `checkpoints/tinymem/mult_clean_ep100.pth` | `6ec92609396ed8d9315681d0a0f2acde` |

</details>

TinyMem models and datasets come from the TinyMem benchmark of
[Sakarvadia et al. (2025)](https://arxiv.org/abs/2410.02159). To train new TinyMem models, use
`src/memorization_in_toy_models.py`; see [docs/TinyMem_original_README.md](docs/TinyMem_original_README.md).

**Reference models.** GSS needs a reference with the same tokenizer, competent in the same
domain, and with low exposure to the evaluated sequences:

| setting | references used |
|---|---|
| Math / Noise | `math_backdoor.pth` (`xbd`), `mult_clean_ep500.pth` (`clean500`), `mult_clean_ep100.pth` (`clean100`) |
| Math / Backdoor | `math_noise.pth` (`xnoise`), `clean500`, `clean100` |
| Language / Noise | `lm_backdoor.pth` (`xbd`) |
| Language / Backdoor | `lm_noise.pth` (`xnoise`) |

Each "sibling" model was trained on the same domain but never saw the noised (respectively
backdoored) sequences of the other setting.

## Reproducing the TinyMem results

The fitted configuration of each setting is stored in
[`configs/gss_tinymem_best.json`](configs/gss_tinymem_best.json). One command fits GSS
(directions + gate threshold) and evaluates it. The steering hook stays active for every
reported metric:

```bash
python scripts/gss_tinymem/run_gss_tinymem.py --setting math_noise
python scripts/gss_tinymem/run_gss_tinymem.py --setting math_backdoor
python scripts/gss_tinymem/run_gss_tinymem.py --setting lm_noise
python scripts/gss_tinymem/run_gss_tinymem.py --setting lm_backdoor
```

Each run prints the unmodified baseline and the GSS numbers as JSON:

- `paper_mem`: memorization in the TinyMem metric (one value per duplication class, or the attack success rate).
- `test_acc` / `test_ppl`: utility on the clean test sets.
- `gate_clean_test`: fraction of clean test tokens at which the gate fires.

Each run takes 7–15 seconds on one NVIDIA L40S.

`configs/gss_tinymem_best.json` holds two selections per setting: `paper` (selected on the
evaluation metrics above, rule "memorization exactly 0, then best utility"; used by default)
and `heldout` (selected on disjoint validation splits; `--protocol heldout`).

Optional sanity check: `scripts/gss_tinymem/verify_fast_metric.py` confirms, row by row, that
the single-forward memorization metric used by the search equals the generation-based TinyMem
metric, with and without steering.

## Hyper-parameter search

`scripts/gss_tinymem/search.py` searches over the reference model, the high-signal threshold
τ on ω, the hooked module, the subspace rank, the gate quantile, the steering strength, the
steering sign, and the consecutive-exceedance requirement *r*. It has three stages:

1. **Stage A (no generation).** Builds the directions for every (reference, τ, module, rank)
   and measures gate selectivity: how often the gate fires right before a memorized token,
   versus how often it fires on clean tokens.
2. **Stage B.** Sweeps the gate quantile, strength, sign and *r* for the most selective
   configurations, and measures memorization and utility.
3. **Stage C.** Fits the selected configuration and evaluates it.

Every result line is written to a `jsonl` file with a unique key, so interrupted runs resume.

```bash
# protocol "paper": select on the TinyMem evaluation metrics
python scripts/gss_tinymem/search.py --setting math_backdoor --protocol paper --tag _paper --top_a 10
# protocol "heldout": select on validation splits, report once on test
python scripts/gss_tinymem/search.py --setting math_backdoor --top_a 10
# write configs/gss_tinymem_best.json and results/gss_tinymem_search/SUMMARY.md
python scripts/gss_tinymem/collect_best.py
```

`--kinds whitened,contrastive` also searches the contrastive construction (paper Eq. 9).
The released configurations use the whitened construction (Eq. 7–8).

Memorization during the search is computed with one teacher-forced forward pass: greedy
decoding reproduces a target if and only if the argmax at every position of the target window
equals the next target token. The model and the steering hook are causal, and EOS is masked
exactly as `generate(min_length=max_length)` does, so the result is identical to generation
while avoiding ~100 decoding steps per sequence.

## Using GSS on your own model

```python
from src import gss_tinymem as gss

cfg = gss.GSSConfig(point="transformer.h.2.mlp", kind="whitened", tau=0.0, rank=1,
                    q=0.9995, beta=0.3, sign=-1.0, r=2)
# calib_rows: [N, T] token ids of memorized sequences; clean_rows: [M, T] clean token ids
artifact = gss.fit(target_model, reference_model, calib_rows, clean_rows, cfg)
hook = gss.attach(target_model, artifact)    # steering is now active for every forward pass
...                                           # generate / evaluate as usual
hook.remove()                                 # detach
```

`point` is the name of any module in `model.named_modules()` whose output is a hidden state
`[batch, time, d]`.

## Method summary

For calibration sequences x, GSS computes the token-level signal

  ω_t = log p_θ(x_t | x_<t) − log p_ref(x_t | x_<t),

and marks high-signal positions (ω_t > τ) and low-signal positions (ω_t ≤ 0). With h_t the
activation at the hooked module and g_t the gradient of the masked cross-entropy:

- **whitened**: M_hi = Σ_hi h_t g_tᵀ and C = Σ_lo + γI = LLᵀ, where Σ_lo is the uncentered
  second moment of the low-signal activations. The SVD L⁻¹M_hi = ŨSṼᵀ gives the probes
  U = L⁻ᵀŨ_{:,1:K} and the steers V = Ṽ_{:,1:K}. This is the closed-form optimum of
  max tr(UᵀM_hiV) s.t. UᵀCU = I, VᵀV = I (Theorem 4.1).
- **gate**: ε is the q-quantile of |u_kᵀh| on clean text. At inference,

  h′_t = h_t − sign · β · Σ_k 1(|u_kᵀh_t| > ε) (u_kᵀh_t) v_k,

  optionally applied only after r consecutive positions exceed ε.

<p align="center">
  <img src="assets/gss_gating.png" alt="Fixed vs gated steering" width="92%">
  <br>
  <sub>Fig. 3: (a) a fixed steering update moves every activation; (b) GSS corrects only above the threshold. Regions are schematic.</sub>
</p>

The steering sign is a hyper-parameter. To first order, the subtractive update
(sign = +1) lowers the cross-entropy at high-signal positions (paper Appendix A.3), so both
signs are searched.

## Repository structure

```
src/gss_tinymem.py                  GSS: statistics, directions, gate calibration, steering hook
scripts/gss_tinymem/common.py       TinyMem settings, data splits, metrics
scripts/gss_tinymem/run_gss_tinymem.py   fit + evaluate one setting
scripts/gss_tinymem/search.py       hyper-parameter search (stages A/B/C, resumable)
scripts/gss_tinymem/collect_best.py selection rule -> configs/gss_tinymem_best.json
scripts/gss_tinymem/verify_fast_metric.py   fast metric == generation metric check
configs/gss_tinymem_best.json       released configurations
results/gss_tinymem_search/         search logs (jsonl) and SUMMARY.md
src/localize/, src/memorization_in_toy_models.py, ...   TinyMem training and baselines (Sakarvadia et al.)
```

## Acknowledgements

This repository builds on the TinyMem benchmark and the memorization-mitigation baselines of
[Sakarvadia et al., *Mitigating Memorization in Language Models* (ICLR 2025)](https://github.com/msakarvadia/memorization);
their original README is kept in [docs/TinyMem_original_README.md](docs/TinyMem_original_README.md).

## Citation

```bibtex
@inproceedings{zhang2026gss,
  title     = {Mitigating Memorization Where It Happens},
  author    = {Zhang, Xuanqi and Shang, Haoyang and Li, Xiaoxiao},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=hV2JIz6mgg},
  eprint    = {2602.08901},
  archivePrefix = {arXiv}
}
```

## License

MIT; see [LICENSE](LICENSE). The TinyMem code and baselines are © 2024 Mansi Sakarvadia
under the same license.
