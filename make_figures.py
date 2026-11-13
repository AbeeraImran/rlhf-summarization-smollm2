import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_rows(path):
    rows = []
    for line in open(path):
        parts = line.strip().split(',')
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            pass
    return rows

def col(rows, i):
    return [r[i] for r in rows]

sft = read_rows('results/sft_log.csv')
rew = read_rows('results/reward_log.csv')
ppo = read_rows('results/ppo_log.csv')
dpo = read_rows('results/dpo_log.csv')
print('rows:', len(sft), len(rew), len(ppo), len(dpo))

plt.figure(figsize=(6, 4))
plt.plot(col(sft, 0), col(sft, 1), marker='o', label='train loss')
plt.plot(col(sft, 0), col(sft, 2), marker='o', label='validation loss')
plt.xlabel('step')
plt.ylabel('loss')
plt.title('SFT loss')
plt.legend()
plt.tight_layout()
plt.savefig('results/fig1_sft_loss.png', dpi=150)
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(col(rew, 0), col(rew, 1), marker='o')
ax[0].set_xlabel('step')
ax[0].set_ylabel('train loss')
ax[0].set_title('Reward model: training loss')
ax[1].plot(col(rew, 0), col(rew, 2), marker='o')
ax[1].axhline(0.5, linestyle='--', color='gray')
ax[1].set_xlabel('step')
ax[1].set_ylabel('validation accuracy')
ax[1].set_title('Reward model: validation accuracy')
plt.tight_layout()
plt.savefig('results/fig2_reward_model.png', dpi=150)
plt.close()

fig, ax = plt.subplots(2, 2, figsize=(10, 7))
names = ['mean reward score', 'mean KL to reference', 'mean summary length (tokens)', 'clip fraction']
for k in range(4):
    a = ax[k // 2][k % 2]
    a.plot(col(ppo, 0), col(ppo, k + 1))
    a.set_xlabel('iteration')
    a.set_ylabel(names[k])
    a.set_title(names[k])
plt.tight_layout()
plt.savefig('results/fig3_ppo.png', dpi=150)
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(col(dpo, 0), col(dpo, 1), marker='o')
ax[0].set_xlabel('step')
ax[0].set_ylabel('train loss')
ax[0].set_title('DPO: training loss')
ax[1].plot(col(dpo, 0), col(dpo, 3), marker='o')
ax[1].axhline(0.5, linestyle='--', color='gray')
ax[1].set_xlabel('step')
ax[1].set_ylabel('implicit reward accuracy')
ax[1].set_title('DPO: validation implicit reward accuracy')
plt.tight_layout()
plt.savefig('results/fig4_dpo.png', dpi=150)
plt.close()
print('figures saved')
