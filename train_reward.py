import sys
sys.path.append('starter')
import os
import time
import torch
import torch.nn as nn
from functools import partial
from torch.utils.data import DataLoader
from models import load_tokenizer, load_causal_lm
from data import load_preference_splits
from batching import collate_pref

class RewardModel(nn.Module):
    def __init__(self, path):
        super().__init__()
        lm = load_causal_lm(path)
        device = next(lm.parameters()).device
        self.body = lm.model.float()
        self.head = nn.Linear(lm.config.hidden_size, 1).to(device)

    def forward(self, input_ids, attention_mask):
        hidden = self.body(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(dim=1) - 1
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        final = hidden[rows, last]
        return self.head(final).squeeze(-1)

def pair_rewards(model, batch, device):
    r_chosen = model(batch["chosen_ids"].to(device), batch["chosen_mask"].to(device))
    r_rejected = model(batch["rejected_ids"].to(device), batch["rejected_mask"].to(device))
    return r_chosen, r_rejected

def bt_loss(r_chosen, r_rejected):
    return -torch.log(torch.sigmoid(r_chosen - r_rejected)).mean()

def accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            r_chosen, r_rejected = pair_rewards(model, batch, device)
            correct += (r_chosen > r_rejected).sum().item()
            total += r_chosen.shape[0]
    model.train()
    return correct / total

def save_reward(model, folder):
    os.makedirs(folder, exist_ok=True)
    torch.save(model.state_dict(), folder + '/reward.pt')

def load_reward(folder, sft_path='checkpoints/sft'):
    model = RewardModel(sft_path)
    device = next(model.parameters()).device
    model.load_state_dict(torch.load(folder + '/reward.pt', map_location=device))
    return model

if __name__ == "__main__":
    torch.manual_seed(42)
    os.makedirs('results', exist_ok=True)

    tok = load_tokenizer()
    pref_train, pref_val = load_preference_splits(tok)
    model = RewardModel('checkpoints/sft')
    device = next(model.parameters()).device

    collate = partial(collate_pref, pad_id=tok.pad_token_id)
    train_loader = DataLoader(pref_train, batch_size=8, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(pref_val, batch_size=8, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    print("val accuracy before training:", accuracy(model, val_loader, device))

    log = open('results/reward_log.csv', 'w')
    log.write("step,train_loss,val_acc\n")

    model.train()
    step = 0
    running = 0
    start = time.time()

    for batch in train_loader:
        optimizer.zero_grad()
        batch_loss = 0
        for half in range(2):
            a = half * 4
            b = a + 4
            part = {
                "chosen_ids": batch["chosen_ids"][a:b],
                "chosen_mask": batch["chosen_mask"][a:b],
                "rejected_ids": batch["rejected_ids"][a:b],
                "rejected_mask": batch["rejected_mask"][a:b],
            }
            r_chosen, r_rejected = pair_rewards(model, part, device)
            loss = bt_loss(r_chosen, r_rejected) / 2
            loss.backward()
            batch_loss += loss.item()
        optimizer.step()

        step += 1
        running += batch_loss

        if step % 250 == 0:
            train_loss = running / 250
            val_acc = accuracy(model, val_loader, device)
            running = 0
            print("step", step, "train loss", round(train_loss, 4), "val acc", round(val_acc, 4))
            log.write(str(step) + "," + str(train_loss) + "," + str(val_acc) + "\n")
            log.flush()

    log.close()
    save_reward(model, 'checkpoints/reward')

    print("total steps:", step)
    print("training time (minutes):", round((time.time() - start) / 60, 2))
