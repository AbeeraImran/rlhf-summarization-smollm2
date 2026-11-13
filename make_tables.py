import os
import csv

def read_rows(path):
    rows = []
    for line in open(path):
        parts = line.strip().split(',')
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            pass
    return rows

def mean(x):
    return sum(x) / len(x)

def md(header, body):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    for r in body:
        out.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(out)

sft = read_rows('results/sft_log.csv')
rew = read_rows('results/reward_log.csv')
ppo = read_rows('results/ppo_log.csv')
dpo = read_rows('results/dpo_log.csv')

last = ppo[-10:]
ppo_score = mean([r[1] for r in last])
ppo_loss = mean([r[5] + 0.1 * r[6] - 0.01 * r[7] for r in last])

t1 = [
    ['SFT train / validation pairs', '20,000 / 500'],
    ['Preference train / validation pairs', '20,000 / 1,000'],
    ['Evaluation posts', '300'],
    ['Mean / max prompt length, SFT train (tokens)', '317.42 / 448'],
    ['Mean / max summary length, SFT train (tokens)', '33.82 / 48'],
]

t2 = [
    ['SFT', round(sft[-1][1], 4), 'val. loss: ' + str(round(sft[-1][2], 4)), '75.87 min, T4'],
    ['Reward model', round(rew[-1][1], 4), 'pairwise acc.: ' + str(round(rew[-1][2], 4)), '116.92 min, T4'],
    ['PPO', round(ppo_loss, 4), 'final mean r (last 10 iterations): ' + str(round(ppo_score, 4)), '48.98 min, T4 (200 iterations)'],
    ['DPO', round(dpo[-1][1], 4), 'implicit reward acc.: ' + str(round(dpo[-1][3], 4)), '72.08 min, T4 (1073 of 2500 steps)'],
]

text = 'Table 1 - Data\n\n' + md(['Item', 'Value'], t1)
text += '\n\nTable 2 - Training\n\n' + md(['Stage', 'Final train loss', 'Validation metric', 'Time / GPU'], t2)

if os.path.exists('results/table3.csv'):
    rows = list(csv.reader(open('results/table3.csv')))
    text += '\n\nTable 3 - Final comparison (300 evaluation posts)\n\n' + md(rows[0], rows[1:])

open('results/tables.md', 'w').write(text)
print(text)
