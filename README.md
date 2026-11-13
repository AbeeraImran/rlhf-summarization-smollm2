# RLHF summarization on SmolLM2-135M (SFT, reward model, PPO, DPO)

Reproduction of the pipeline from "Learning to Summarize from Human Feedback" (Stiennon et al., 2020) on HuggingFaceTB/SmolLM2-135M with Reddit TL;DR data. All losses and training loops are written in plain PyTorch (no trl, peft, trlx or Trainer). Single author.

## Files
- starter/: given files, unchanged
- common.py: token_logprobs, get_summary_mask, sft_loss, generate_summaries
- train_sft.py, train_reward.py, train_ppo.py, train_dpo.py, evaluate.py
- make_figures.py, make_tables.py, make_samples.py, make_human.py
- results/: logs, tables.md, table3.csv, eval_summaries.jsonl, human_eval.csv, figures, sft_before_after.json

## How to run
pip install torch transformers datasets rouge-score matplotlib
python starter/check_starter.py
python train_sft.py
python train_reward.py
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_dpo.py 2500 400 70
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_ppo.py 200 50 8
python evaluate.py
python make_figures.py
python make_tables.py

Run on a Colab T4. The last three numbers of train_dpo.py are: maximum steps, validation pairs for the intermediate checks, and a time budget in minutes. For train_ppo.py: iterations, time budget in minutes, chunk size.

Checkpoints are not in this repository. They are on Google Drive: https://drive.google.com/drive/folders/1ip6FUtHObFNJLZME6DziuA_ONJWjaWeL?usp=sharing.

## Deviations and notes
- DPO stopped at its 70 minute budget after 1073 of the 2500 steps (about 43% of one epoch). The 250-step checks used the first 400 validation pairs; the final check used all 1000.
- Batches that did not fit on the T4 (SFT 16, reward model 8, DPO 8, PPO rollout of 16) were split into halves or chunks and the gradients were added before optimizer.step().
- Reward model, PPO and DPO models run in float32. No gradient clipping for the reward model or DPO (not specified in the assignment).
- PPO uses the exact settings in the assignment (clip 0.2, KL 0.05, value 0.1, entropy 0.01, gamma 1, 4 epochs, lr 1e-5, grad clip 1.0, 16 prompts, 200 iterations).
- Summary length in tables includes the end-of-sequence token. ROUGE-L uses stemming.
- Human evaluation: single member, no second rater. The 30 posts were shown on a blind sheet (labels shuffled, seed 42).
- Front end: not included in this submission.

## Results
Tables 1 to 3 are in results/tables.md.

### Table 1 - Data
| Item | Value |
|---|---|
| SFT train / validation pairs | 20,000 / 500 |
| Preference train / validation pairs | 20,000 / 1,000 |
| Evaluation posts | 300 |
| Mean / max prompt length, SFT train (tokens) | 317.42 / 448 |
| Mean / max summary length, SFT train (tokens) | 33.82 / 48 |

### Table 2 - Training
| Stage | Final train loss | Validation metric | Time / GPU |
|---|---|---|---|
| SFT | 2.4249 | val. loss 2.4497 | 75.87 min, T4 |
| Reward model | 0.5986 | pairwise acc. 0.614 | 116.92 min, T4 |
| PPO | 0.0044 (total PPO loss, advantages normalised to mean 0) | final mean r (last 10 iterations) 4.9531 | 48.98 min, T4 (200 iterations) |
| DPO | 0.6415 | implicit reward acc. 0.608 | 72.08 min, T4 (1073 of 2500 steps) |

### Table 3 - Final comparison (300 evaluation posts)
| Policy | Mean r | Win rate vs. SFT | KL to SFT | Length | ROUGE-L |
|---|---|---|---|---|---|
| SFT | 4.3007 | NA | 0 | 30.72 | 0.2541 |
| PPO | 4.7502 | 0.5867 | 11.9796 | 22.5 | 0.193 |
| DPO | 3.4927 | 0.4533 | 2.8172 | 44.32 | 0.2246 |

### Table 4 - Human evaluation (30 posts)
| | SFT picked | PPO picked | DPO picked |
|---|---|---|---|
| Member 1 | 14 | 1 | 15 |
| Member 2 | not applicable (single member) | | |
| Agreement between members | not applicable | | |

## Figures
- results/fig1_sft_loss.png, results/fig2_reward_model.png, results/fig3_ppo.png, results/fig4_dpo.png

## Correctness checks (outputs from my runs)
- SFT mask: decoding the tokens where the summary mask is 1 gave the summary text followed by the end-of-sequence token (check passed).
- Reward model padding: the same row scored -0.35965 in a padded batch and -0.35965 alone.
- Frozen models: `reference frozen: True` and `reward model frozen: True` in PPO; `reference frozen: True` in DPO.
- PPO first iteration: KL to the reference = 0.0.
- PPO every iteration, epoch 1: max |ratio - 1| = 0.0 and max clip fraction = 0.0 over all 200 iterations.
- DPO first step: loss 0.6931 (log 2 = 0.6931).
