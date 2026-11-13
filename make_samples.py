import json

rows = [json.loads(line) for line in open('results/eval_summaries.jsonl')]
chosen = []

def diff(r, m):
    return r[m]['reward'] - r['sft']['reward']

def take(order, k):
    n = 0
    for r in order:
        if n == k:
            break
        if r['index'] in [c['index'] for c in chosen]:
            continue
        chosen.append(r)
        n += 1

by_ppo = sorted(rows, key=lambda r: diff(r, 'ppo'), reverse=True)
by_dpo = sorted(rows, key=lambda r: diff(r, 'dpo'), reverse=True)
take(by_ppo, 2)
take(by_ppo[::-1], 2)
take(by_dpo, 2)
take(by_dpo[::-1], 2)
take(rows[::37], 2)

out = []
for r in chosen:
    out.append('POST ' + str(r['index']))
    out.append('POST END: ' + r['prompt'][-250:].replace('\n', ' '))
    out.append('REFERENCE: ' + r['reference'])
    for m in ['sft', 'ppo', 'dpo']:
        out.append(m.upper() + ' (reward ' + str(r[m]['reward']) + '): ' + r[m]['summary'])
    out.append('')
open('results/samples_draft.md', 'w').write('\n'.join(out))
print('\n'.join(out))
