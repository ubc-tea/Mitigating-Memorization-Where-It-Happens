# GSS on TinyMem: final configurations

Construction: whitened only. Selection: memorization exactly 0, then best utility. The steering hook is active for every metric.

## Protocol `paper` (hyper-parameters tuned on the test metrics)

| setting | paper Ours | baseline (ours) | GSS mem | GSS utility | gate on clean test tokens | config |
|---|---|---|---|---|---|---|
| math_noise | 0.00 / Acc 96.98 | 34.55 / Acc 97.00 | 0.00 | Acc 91.37 | 10.917% | ref=clean500 h.3 tau=0 K=2 q=0.95 beta=1 sign=-1 r=1 |
| math_backdoor | 0.00 / Acc 96.82 | 99.55 / Acc 97.01 | 0.00 | Acc 97.01 | 0.003% | ref=xnoise h.2.mlp tau=0 K=1 q=0.9995 beta=0.3 sign=-1 r=2 |
| lm_noise | 0.00 / PPL 63.13 | 15.20 / PPL 57.60 | 0.00 | PPL 57.68 | 0.011% | ref=xbd h.2.mlp tau=1 K=1 q=0.99 beta=10 sign=+1 r=2 |
| lm_backdoor | 0.00 / PPL 63.17 | 100.00 / PPL 65.20 | 0.00 | PPL 68.84 | 0.178% | ref=xnoise h.0 tau=3 K=4 q=0.999 beta=1 sign=+1 r=1 |

## Protocol `heldout` (selected on validation; test only where already evaluated)

| setting | val mem | val utility | test mem (held-out) | test utility | config |
|---|---|---|---|---|---|
| math_noise | 0.00 | Acc 94.34 | 3.77 | Acc 94.18 | ref=clean500 h.3 tau=1 K=4 q=0.99 beta=1 sign=+1 r=1 |
| math_backdoor | 0.00 | Acc 97.29 | 0.00 | Acc 97.01 | ref=xnoise h.2.mlp tau=1 K=1 q=0.995 beta=0.3 sign=-1 r=2 |
| lm_noise | 0.00 | PPL 66.50 | not evaluated | – | ref=xbd h.2.mlp tau=3 K=2 q=0.999 beta=0.3 sign=+1 r=1 |
| lm_backdoor | 0.00 | PPL 67.18 | not evaluated | – | ref=xnoise h.1 tau=0 K=1 q=0.95 beta=10 sign=-1 r=2 |

Memorization: noise settings = % of noised rows reproduced (max over duplication classes); backdoor = attack success on the poisoned test set (paper) / poisoned held-out train rows (heldout val).
