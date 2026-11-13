import os
import sys
import csv
import time
import torch
import torch.nn as nn
from functools import partial
from torch.utils.data import DataLoader
sys.path.insert(0, 'starter')
from config import SEED
from models import load_tokenizer, load_causal_lm
from data import load_sft_splits
from batching import collate_prompts, _pad
from train_reward import load_reward

ITERS = int(sys.argv[1])
BUDGET = float(sys.argv[2])
CHUNK = int(sys.argv[3])
EPS = 0.2
BETA = 0.05
CV = 0.1
CH = 0.01
EPOCHS = 4
LR = 1e-5
CLIP = 1.0
BATCH = 16
MAX_NEW = 48
device = 'cuda'

def forward_gen(model, head, ids, mask, P):
    hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
    h = hidden[:, P - 1:-1]
    logp_all = torch.log_softmax(model.lm_head(h), dim=-1)
    logp = logp_all.gather(2, ids[:, P:].unsqueeze(-1)).squeeze(-1)
    entropy = -(logp_all.exp() * logp_all).sum(dim=-1)
    value = None
    if head is not None:
        value = head(h).squeeze(-1)
    return logp, value, entropy

def chunk_pass(model, head, ids, mask, P):
    lps = []
    vals = []
    with torch.no_grad():
        for a in range(0, ids.shape[0], CHUNK):
            lp, v, _ = forward_gen(model, head, ids[a:a + CHUNK], mask[a:a + CHUNK], P)
            lps.append(lp)
            if head is not None:
                vals.append(v)
    if head is None:
        return torch.cat(lps), None
    return torch.cat(lps), torch.cat(vals)

def score(reward, ids, mask, pad):
    seqs = [ids[i][mask[i].bool()].tolist() for i in range(ids.shape[0])]
    r_ids, r_mask = _pad(seqs, pad, 'right')
    with torch.no_grad():
        return reward(r_ids.to(device), r_mask.to(device))

torch.manual_seed(SEED)
os.makedirs('checkpoints/ppo', exist_ok=True)
os.makedirs('results', exist_ok=True)

tok = load_tokenizer()
pad = tok.pad_token_id
eos = tok.eos_token_id
sft_train, _ = load_sft_splits(tok)
g = torch.Generator().manual_seed(SEED)
loader = DataLoader(sft_train, batch_size=BATCH, shuffle=True, collate_fn=partial(collate_prompts, pad_id=pad), generator=g, drop_last=True)

policy = load_causal_lm('checkpoints/sft').float()
ref = load_causal_lm('checkpoints/sft').float()
reward = load_reward('checkpoints/reward')
for frozen in (ref, reward):
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad = False
head = nn.Linear(policy.config.hidden_size, 1).to(device)
nn.init.zeros_(head.weight)
nn.init.zeros_(head.bias)
print('reference frozen:', all(not p.requires_grad for p in ref.parameters()))
print('reward model frozen:', all(not p.requires_grad for p in reward.parameters()))

params = list(policy.parameters()) + list(head.parameters())
opt = torch.optim.AdamW(params, lr=LR)

with open('results/ppo_log.csv', 'w', newline='') as f:
    csv.writer(f).writerow(['iteration', 'score', 'kl', 'length', 'clip_frac', 'policy_loss', 'value_loss', 'entropy', 'ep1_clip', 'ep1_ratio_dev', 'minutes'])

start = time.time()
it = 0
worst_dev = 0.0
worst_clip = 0.0

for batch in loader:
    it += 1
    pid = batch['input_ids'].to(device)
    pmask = batch['attention_mask'].to(device)
    B = pid.shape[0]
    P = pid.shape[1]

    with torch.no_grad():
        out = policy.generate(input_ids=pid, attention_mask=pmask, do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=MAX_NEW, pad_token_id=pad, eos_token_id=eos)
    gen = out[:, P:]
    is_eos = gen == eos
    first = is_eos.int().argmax(dim=1)
    T = torch.where(is_eos.any(dim=1), first + 1, torch.full_like(first, gen.shape[1]))
    Tm = int(T.max().item())
    am = (torch.arange(Tm, device=device).unsqueeze(0) < T.unsqueeze(1)).long()
    ids = torch.cat([pid, gen[:, :Tm]], dim=1)
    mask = torch.cat([pmask, am], dim=1)

    scores = score(reward, ids, mask, pad)
    old_lp, old_v = chunk_pass(policy, head, ids, mask, P)
    ref_lp, _ = chunk_pass(ref, None, ids, mask, P)

    mt = am.float()
    N = mt.sum().item()
    diff = (old_lp - ref_lp) * mt
    rew = -BETA * diff
    rew[torch.arange(B, device=device), T - 1] += scores
    ret = rew.flip(1).cumsum(1).flip(1)
    adv = (ret - old_v) * mt
    mean = adv.sum() / N
    std = torch.sqrt((((adv - mean) * mt) ** 2).sum() / N)
    adv = (adv - mean) / (std + 1e-8) * mt

    kl = diff.sum(dim=1).mean().item()
    pl_sum = 0.0
    vl_sum = 0.0
    en_sum = 0.0
    clip_sum = 0.0
    ep1_clip = 0.0
    ep1_dev = 0.0

    for epoch in range(EPOCHS):
        opt.zero_grad()
        e_pl = 0.0
        e_vl = 0.0
        e_en = 0.0
        e_clip = 0.0
        e_dev = 0.0
        for a in range(0, B, CHUNK):
            sl = slice(a, a + CHUNK)
            lp, v, ent = forward_gen(policy, head, ids[sl], mask[sl], P)
            m = mt[sl]
            ratio = torch.exp(lp - old_lp[sl])
            s1 = ratio * adv[sl]
            s2 = torch.clamp(ratio, 1 - EPS, 1 + EPS) * adv[sl]
            pl = -(torch.min(s1, s2) * m).sum() / N
            vl = (((v - ret[sl]) ** 2) * m).sum() / N
            en = (ent * m).sum() / N
            loss = pl + CV * vl - CH * en
            loss.backward()
            e_pl += pl.item()
            e_vl += vl.item()
            e_en += en.item()
            dev = (ratio.detach() - 1).abs() * m
            e_clip += ((dev > EPS).float() * m).sum().item()
            e_dev = max(e_dev, dev.max().item())
        torch.nn.utils.clip_grad_norm_(params, CLIP)
        opt.step()
        e_clip = e_clip / N
        if epoch == 0:
            ep1_clip = e_clip
            ep1_dev = e_dev
        pl_sum += e_pl
        vl_sum += e_vl
        en_sum += e_en
        clip_sum += e_clip

    worst_dev = max(worst_dev, ep1_dev)
    worst_clip = max(worst_clip, ep1_clip)
    minutes = (time.time() - start) / 60
    row = [it, scores.mean().item(), kl, T.float().mean().item(), clip_sum / EPOCHS, pl_sum / EPOCHS, vl_sum / EPOCHS, en_sum / EPOCHS, ep1_clip, ep1_dev, minutes]
    with open('results/ppo_log.csv', 'a', newline='') as f:
        csv.writer(f).writerow([round(x, 6) if isinstance(x, float) else x for x in row])
    print('iter', it, 'score', round(row[1], 4), 'kl', round(kl, 4), 'len', round(row[3], 1), 'clip', round(row[4], 4), 'ep1 clip', round(ep1_clip, 4), 'ep1 dev', round(ep1_dev, 8), 'min', round(minutes, 1))
    if it == 1:
        print('first iteration KL:', kl)
    if it % 25 == 0:
        policy.save_pretrained('checkpoints/ppo')
        tok.save_pretrained('checkpoints/ppo')
    if it == ITERS or minutes > BUDGET:
        break

policy.save_pretrained('checkpoints/ppo')
tok.save_pretrained('checkpoints/ppo')
print('iterations done:', it)
print('max |ratio - 1| in epoch 1 over all iterations:', worst_dev)
print('max clip fraction in epoch 1 over all iterations:', worst_clip)
print('training time (minutes):', round((time.time() - start) / 60, 2))
