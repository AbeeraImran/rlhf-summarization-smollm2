import sys
sys.path.append('starter')
import os
import time
import torch
from functools import partial
from torch.utils.data import DataLoader
from models import load_tokenizer, load_causal_lm
from data import load_sft_splits
from batching import collate_sft
from common import sft_loss, token_logprobs, get_summary_mask

torch.manual_seed(42)
os.makedirs('results', exist_ok=True)

tok = load_tokenizer()
sft_train, sft_val = load_sft_splits(tok)
model = load_causal_lm()
device = next(model.parameters()).device

collate = partial(collate_sft, pad_id=tok.pad_token_id)
train_loader = DataLoader(sft_train, batch_size=16, shuffle=True, collate_fn=collate)
val_loader = DataLoader(sft_val, batch_size=16, shuffle=False, collate_fn=collate)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

def validate():
    model.eval()
    total = 0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            loss = sft_loss(model, ids, mask, batch["prompt_lens"])
            total += loss.item()
            count += 1
    model.train()
    return total / count

print("val loss before training:", validate())

log = open('results/sft_log.csv', 'w')
log.write("step,train_loss,val_loss\n")

model.train()
step = 0
running = 0
start = time.time()

for batch in train_loader:
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    summary_mask = get_summary_mask(ids, mask, batch["prompt_lens"])
    total_tokens = summary_mask.sum()

    optimizer.zero_grad()
    batch_loss = 0
    for half in range(2):
        a = half * 8
        b = a + 8
        logp = token_logprobs(model, ids[a:b], mask[a:b])
        loss = -(logp * summary_mask[a:b]).sum() / total_tokens
        loss.backward()
        batch_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    step += 1
    running += batch_loss

    if step % 250 == 0:
        train_loss = running / 250
        val_loss = validate()
        running = 0
        print("step", step, "train loss", round(train_loss, 4), "val loss", round(val_loss, 4))
        log.write(str(step) + "," + str(train_loss) + "," + str(val_loss) + "\n")
        log.flush()

log.close()

model.save_pretrained('checkpoints/sft')
tok.save_pretrained('checkpoints/sft')

print("total steps:", step)
print("training time (minutes):", round((time.time() - start) / 60, 2))
