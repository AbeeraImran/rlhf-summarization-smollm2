import sys
import json
import csv
import torch
from functools import partial
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
sys.path.insert(0, 'starter')
from models import load_tokenizer, load_causal_lm
from data import load_eval_posts
from batching import collate_prompts, _pad
from common import token_logprobs, get_summary_mask
from train_reward import load_reward

device = 'cuda'
BS = 8
NAMES = ['sft', 'ppo', 'dpo']

tok = load_tokenizer()
pad = tok.pad_token_id
eos = tok.eos_token_id
posts = load_eval_posts(tok)
refs = [p['summary'].strip() for p in posts]
reward = load_reward('checkpoints/reward')
reward.eval()
sft = load_causal_lm('checkpoints/sft').float()
sft.eval()
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def generate_all(model):
    loader = DataLoader(posts, batch_size=BS, shuffle=False, collate_fn=partial(collate_prompts, pad_id=pad))
    outs = []
    for batch in loader:
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=48, pad_token_id=pad, eos_token_id=eos)
        for row in out[:, ids.shape[1]:].tolist():
            if eos in row:
                row = row[:row.index(eos) + 1]
            outs.append(row)
    return outs

def build(summaries, idx):
    seqs = [posts[i]['prompt_ids'] + summaries[i] for i in idx]
    plens = torch.tensor([len(posts[i]['prompt_ids']) for i in idx])
    ids, mask = _pad(seqs, pad, 'right')
    return ids.to(device), mask.to(device), plens

def score_all(summaries):
    vals = []
    for a in range(0, len(posts), BS):
        idx = range(a, min(a + BS, len(posts)))
        ids, mask, _ = build(summaries, idx)
        with torch.no_grad():
            vals.extend(reward(ids, mask).tolist())
    return vals

def logprob_sums(model, summaries):
    vals = []
    for a in range(0, len(posts), BS):
        idx = range(a, min(a + BS, len(posts)))
        ids, mask, plens = build(summaries, idx)
        with torch.no_grad():
            lp = token_logprobs(model, ids, mask)
        m = get_summary_mask(ids, mask, plens)
        vals.extend((lp * m).sum(dim=1).tolist())
    return vals

def mean(x):
    return sum(x) / len(x)

results = {}
for name in NAMES:
    if name == 'sft':
        model = sft
    else:
        model = load_causal_lm('checkpoints/' + name).float()
        model.eval()
    summaries = generate_all(model)
    rewards = score_all(summaries)
    kls = [a - b for a, b in zip(logprob_sums(model, summaries), logprob_sums(sft, summaries))]
    texts = [tok.decode(s, skip_special_tokens=True).strip() for s in summaries]
    rouge = [scorer.score(refs[i], texts[i])['rougeL'].fmeasure for i in range(len(posts))]
    results[name] = {'text': texts, 'reward': rewards, 'kl': kls, 'length': [len(s) for s in summaries], 'rouge': rouge}
    print(name, 'done')
    if name != 'sft':
        del model
        torch.cuda.empty_cache()

with open('results/eval_summaries.jsonl', 'w') as f:
    for i in range(len(posts)):
        row = {'index': i, 'prompt': posts[i]['prompt'], 'reference': refs[i]}
        for name in NAMES:
            r = results[name]
            row[name] = {'summary': r['text'][i], 'reward': round(r['reward'][i], 4), 'kl': round(r['kl'][i], 4), 'length': r['length'][i], 'rougeL': round(r['rouge'][i], 4)}
        f.write(json.dumps(row) + '\n')

rows = []
for name in NAMES:
    r = results[name]
    if name == 'sft':
        win = 'NA'
    else:
        win = round(mean([1.0 if a > b else 0.0 for a, b in zip(r['reward'], results['sft']['reward'])]), 4)
    rows.append([name, round(mean(r['reward']), 4), win, round(mean(r['kl']), 4), round(mean(r['length']), 2), round(mean(r['rouge']), 4)])

with open('results/table3.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['policy', 'mean_reward', 'win_rate_vs_sft', 'kl_to_sft', 'length', 'rougeL'])
    w.writerows(rows)

print('policy | mean reward | win rate vs SFT | KL to SFT | length | ROUGE-L')
for r in rows:
    print(' | '.join(str(x) for x in r))
