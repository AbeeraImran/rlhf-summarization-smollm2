Table 1 - Data

| Item | Value |
|---|---|
| SFT train / validation pairs | 20,000 / 500 |
| Preference train / validation pairs | 20,000 / 1,000 |
| Evaluation posts | 300 |
| Mean / max prompt length, SFT train (tokens) | 317.42 / 448 |
| Mean / max summary length, SFT train (tokens) | 33.82 / 48 |

Table 2 - Training

| Stage | Final train loss | Validation metric | Time / GPU |
|---|---|---|---|
| SFT | 2.4249 | val. loss: 2.4497 | 75.87 min, T4 |
| Reward model | 0.5986 | pairwise acc.: 0.614 | 116.92 min, T4 |
| PPO | 0.0044 | final mean r (last 10 iterations): 4.9531 | 48.98 min, T4 (200 iterations) |
| DPO | 0.6415 | implicit reward acc.: 0.608 | 72.08 min, T4 (1073 of 2500 steps) |