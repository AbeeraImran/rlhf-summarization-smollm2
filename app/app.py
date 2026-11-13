import os
import sys
import torch
import gradio as gr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE + '/starter')
sys.path.insert(0, BASE)

from models import load_tokenizer, load_causal_lm
from batching import _pad
from train_reward import load_reward

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = load_tokenizer()
pad = tok.pad_token_id
eos = tok.eos_token_id

models = {}
for name in ['sft', 'ppo', 'dpo']:
    m = load_causal_lm(BASE + '/checkpoints/' + name).float()
    m.eval()
    models[name] = m
reward = load_reward(BASE + '/checkpoints/reward', BASE + '/checkpoints/sft')
reward.eval()

EXAMPLE = 'SUBREDDIT: r/relationships\nTITLE: My boyfriend never plans anything\nPOST: We have been together for two years. I always plan the dates, the trips and even our anniversary. When I told him it makes me feel unappreciated he said he is just bad at planning. I do not want to break up but I am tired of doing everything. What should I do?'

def build_prompt(text):
    post = text.split('TL;DR:')[0].strip()
    post_ids = tok(post, add_special_tokens=False)['input_ids'][:440]
    post = tok.decode(post_ids)
    return tok(post + '\nTL;DR:', add_special_tokens=False)['input_ids']

def summarize(model, prompt_ids):
    ids = torch.tensor([prompt_ids]).to(device)
    mask = torch.ones_like(ids)
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=48, pad_token_id=pad, eos_token_id=eos)
    new = out[0, ids.shape[1]:].tolist()
    if eos in new:
        new = new[:new.index(eos) + 1]
    r_ids, r_mask = _pad([prompt_ids + new], pad, 'right')
    with torch.no_grad():
        score = reward(r_ids.to(device), r_mask.to(device)).item()
    return tok.decode(new, skip_special_tokens=True).strip(), round(score, 3)

def run(text):
    if len(text.strip()) == 0:
        return ['', None, '', None, '', None]
    prompt_ids = build_prompt(text)
    out = []
    for name in ['sft', 'ppo', 'dpo']:
        s, r = summarize(models[name], prompt_ids)
        out.append(s)
        out.append(r)
    return out

with gr.Blocks() as demo:
    gr.Markdown('# SFT vs PPO vs DPO summaries')
    gr.Markdown('Paste a Reddit post (SUBREDDIT / TITLE / POST format works best).')
    box = gr.Textbox(lines=10, label='Reddit post', value=EXAMPLE)
    btn = gr.Button('Summarize')
    with gr.Row():
        with gr.Column():
            gr.Markdown('### SFT')
            t1 = gr.Textbox(label='summary')
            s1 = gr.Number(label='reward-model score')
        with gr.Column():
            gr.Markdown('### PPO')
            t2 = gr.Textbox(label='summary')
            s2 = gr.Number(label='reward-model score')
        with gr.Column():
            gr.Markdown('### DPO')
            t3 = gr.Textbox(label='summary')
            s3 = gr.Number(label='reward-model score')
    btn.click(run, inputs=box, outputs=[t1, s1, t2, s2, t3, s3])

demo.launch(share=True)
