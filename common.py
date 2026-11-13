import torch

def token_logprobs(model, input_ids, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logp = torch.log_softmax(logits, dim=-1)
    return logp.gather(2, targets.unsqueeze(-1)).squeeze(-1)

def get_summary_mask(input_ids, attention_mask, prompt_lens):
    device = input_ids.device
    L = input_ids.shape[1]
    positions = torch.arange(1, L, device=device).unsqueeze(0)
    is_summary = positions >= prompt_lens.to(device).unsqueeze(1)
    is_real = attention_mask[:, 1:] == 1
    return (is_summary & is_real).long()

def sft_loss(model, input_ids, attention_mask, prompt_lens):
    logp = token_logprobs(model, input_ids, attention_mask)
    mask = get_summary_mask(input_ids, attention_mask, prompt_lens)
    total = (logp * mask).sum()
    return -total / mask.sum()

def generate_summaries(model, tok, input_ids, attention_mask):
    out = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                         max_new_tokens=48, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    new_tokens = out[:, input_ids.shape[1]:]
    texts = []
    for row in new_tokens:
        texts.append(tok.decode(row, skip_special_tokens=True))
    return texts
