import json
import random
import csv

rows = [json.loads(line) for line in open('results/eval_summaries.jsonl')]
rng = random.Random(42)
picked = sorted(rng.sample(range(len(rows)), 30))
names = ['sft', 'ppo', 'dpo']
sheet = []
key = []
for n, i in enumerate(picked):
    order = names[:]
    rng.shuffle(order)
    r = rows[i]
    sheet.append('POST ' + str(n + 1))
    sheet.append(r['prompt'][-600:].replace('\n', ' '))
    for letter, name in zip('ABC', order):
        sheet.append(letter + ') ' + r[name]['summary'])
    sheet.append('=' * 40)
    key.append([n + 1, i, order[0], order[1], order[2]])
open('results/human_eval_sheet.txt', 'w').write('\n'.join(sheet))
with open('results/human_eval_key.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['post', 'eval_index', 'A', 'B', 'C'])
    w.writerows(key)
print('\n'.join(sheet))
