import os
import sys
import csv
import math
import time
import torch
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
sys.path.insert(0, 'starter')
from config import SEED
from models import load_tokenizer, load_causal_lm
from data import load_preference_splits
from batching import collate_pref
from common import token_logprobs, get_summary_mask

STEPS = int(sys.argv[1])
N_VAL = int(sys.argv[2])
BUDGET = float(sys.argv[3])
BETA = 0.1
LR = 2e-6
BATCH = 8
HALF = 4
LOG_EVERY = 250
device = 'cuda'

os.makedirs('checkpoints/dpo', exist_ok=True)
os.makedirs('results', exist_ok=True)
with open('results/dpo_log.csv', 'w', newline='') as f:
    csv.writer(f).writerow(['step', 'train_loss', 'val_loss', 'val_acc', 'n_val'])

def to_dev(b):
    return {k: v.to(device) for k, v in b.items()}

def part(b, i, j):
    return {k: v[i:j] for k, v in b.items()}

def seq_logprob(model, ids, mask, plens):
    lp = token_logprobs(model, ids, mask)
    sm = get_summary_mask(ids, mask, plens)
    return (lp * sm).sum(dim=1)

def pair_logprobs(model, b):
    c = seq_logprob(model, b['chosen_ids'], b['chosen_mask'], b['prompt_lens'])
    r = seq_logprob(model, b['rejected_ids'], b['rejected_mask'], b['prompt_lens'])
    return c, r

def dpo_margin(pc, pr, rc, rr):
    return BETA * ((pc - rc) - (pr - rr))

def evaluate(n_batches):
    policy.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for i in range(n_batches):
            pc, pr = pair_logprobs(policy, val_batches[i])
            rc, rr = val_ref[i]
            m = dpo_margin(pc, pr, rc, rr)
            total_loss += -F.logsigmoid(m).sum().item()
            correct += (m > 0).sum().item()
            total += m.numel()
    policy.train()
    return total_loss / total, correct / total

def write_log(step, tl, vl, va, n):
    with open('results/dpo_log.csv', 'a', newline='') as f:
        csv.writer(f).writerow([step, round(tl, 4), round(vl, 4), round(va, 4), n])

def save():
    policy.save_pretrained('checkpoints/dpo')
    tok.save_pretrained('checkpoints/dpo')

tok = load_tokenizer()
pad = tok.pad_token_id
pref_train, pref_val = load_preference_splits(tok)
collate = partial(collate_pref, pad_id=pad)
g = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(pref_train, batch_size=BATCH, shuffle=True, collate_fn=collate, generator=g)
val_batches = [to_dev(b) for b in DataLoader(pref_val, batch_size=8, shuffle=False, collate_fn=collate)]

policy = load_causal_lm('checkpoints/sft').float()
ref = load_causal_lm('checkpoints/sft').float()
ref.eval()
for p in ref.parameters():
    p.requires_grad = False
print('reference frozen:', all(not p.requires_grad for p in ref.parameters()))
policy.train()

with torch.no_grad():
    val_ref = [pair_logprobs(ref, b) for b in val_batches]
print('val pairs:', sum(b['prompt_lens'].shape[0] for b in val_batches))

opt = torch.optim.AdamW(policy.parameters(), lr=LR)
nb = N_VAL // 8
start = time.time()
run_loss = 0.0
since = 0
step = 0

for b in train_loader:
    b = to_dev(b)
    n = b['prompt_lens'].shape[0]
    opt.zero_grad()
    step_loss = 0.0
    for i in range(0, n, HALF):
        h = part(b, i, i + HALF)
        with torch.no_grad():
            rc, rr = pair_logprobs(ref, h)
        pc, pr = pair_logprobs(policy, h)
        m = dpo_margin(pc, pr, rc, rr)
        loss = -F.logsigmoid(m).sum() / n
        loss.backward()
        step_loss += loss.item()
    opt.step()
    step += 1
    since += 1
    run_loss += step_loss
    if step == 1:
        print('first step loss:', round(step_loss, 4), 'log 2 =', round(math.log(2), 4))
    if step == 10:
        print('seconds per step:', round((time.time() - start) / 10, 2))
    if step % LOG_EVERY == 0:
        vl, va = evaluate(nb)
        write_log(step, run_loss / since, vl, va, nb * 8)
        print('step', step, 'train loss', round(run_loss / since, 4), 'val loss', round(vl, 4), 'val acc', round(va, 4), 'minutes', round((time.time() - start) / 60, 1))
        run_loss = 0.0
        since = 0
        save()
    if step == STEPS or (time.time() - start) / 60 > BUDGET:
        break

tl = run_loss / since if since > 0 else 0.0
vl, va = evaluate(len(val_batches))
write_log(step, tl, vl, va, len(val_batches) * 8)
save()
print('final step', step, 'val loss', round(vl, 4), 'val acc', round(va, 4), 'on all val pairs')
print('training time (minutes):', round((time.time() - start) / 60, 2))
