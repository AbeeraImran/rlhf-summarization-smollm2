# Analysis (Task 7)

All numbers come from my own runs (results/ folder). I worked alone, so the human evaluation has one rater.

## 7.1 Reward model accuracy and misranked pairs

The reward model got 0.522 pairwise accuracy on the 1,000 validation pairs before training (about chance) and 0.614 at the end. The best value was 0.636 at step 2000, so the curve is noisy by a couple of points. With 1,000 pairs the standard error is about 1.5 points, so 0.614 is clearly above chance but this is a weak reward model. Final train loss was 0.5986.

To look at the errors I scored the first 48 of the 1,000 validation pairs on CPU (my GPU quota ran out), so the numbers below are from a small sample and are only a hint. The accuracy on these 48 pairs was 0.6875, which is within noise of the 0.614 on all 1,000 pairs.

Length: when the human-chosen summary was the longer one (31 pairs) the model was right 80.7% of the time. When the chosen summary was the shorter one (17 pairs) it was right only 47.1%, about chance. The reward model picked the longer summary in 70.8% of pairs, while the human labellers picked the longer one in 64.6%. So the model leans towards longer summaries more than the labellers do, and it mostly fails when the better summary is the short one. With 48 pairs this is not a strong result.

Three misranked pairs (reward of the chosen summary is lower than the rejected one):

1. Post: a girl texts the guy she is dating, he does not reply to her meet-up requests. Chosen (4.78): "guy I'm dating initially seems really interested, but then just doesn't respond to my requests for meet-ups. Why? What should I do?" Rejected (5.53): "guy I'm dating initially was very attentive, then I get blown off by his lack of response when i ask him to meet up. why? what should i do/say?" Both are good and almost the same. The rejected one is slightly longer and has more detail, and the reward model scores it higher. The pair is very close in quality, so a human label here is partly a coin flip.
2. Post: a boyfriend who does not make plans and is afraid of losing his girlfriend. Chosen (4.32): "I am far from an emotional person, and my girlfriend is in a position where she's seemingly lost her love for me, and I would like some advice/help." Rejected (4.95): "Girlfriend has seemingly given me short and unpracticed glances in regards to making plans, even though I am an introvert and cannot actively make plans. Help pls??" The rejected one copies more specific details from the post (plans, introvert), which the model likes. The chosen one is vaguer, but the labeller preferred it, probably because it states the real problem.
3. Post: a woman in a long-distance relationship she calls toxic, asking if she can stop answering. Chosen (4.32): "Is it ever okay to not officially end things? emotionally abusive LDR. When is it okay to just stop responding?" Rejected (4.84): "Me [23f] with [23m] 2.5 years LD. In what circumstances is it acceptable to go dark?" The rejected one has the concrete tags (ages, 2.5 years), which look like a typical good TL;DR in the training data. The chosen one captures the emotional abuse, which the rejected one misses. The model seems to reward surface style and concrete details more than what the summary says.

In short, the pairs it gets wrong are close pairs, pairs where the better summary is shorter or vaguer, and pairs where the worse summary has more concrete details or a typical TL;DR format. This matches what PPO exploited in 7.2.

## 7.2 PPO: reward, KL, length, and reward vs quality

Figure 3 (results/fig3_ppo.png) shows the curves. The mean reward-model score rose from 4.24 (first 10 iterations) to 4.95 (last 10), but not smoothly: it dipped to about 4.4 around iterations 61 to 70. The KL to the reference grew fast, from 0 at iteration 1 to about 30 to 43 (last 10 iterations mean about 31). This is the KL on sampled summaries summed over tokens. On the greedy evaluation summaries the KL to SFT is 11.98, so the numbers differ because they are measured on different summaries. The KL did not grow slowly here.

Summary length changed a lot. It was about 34 tokens at the start, fell to about 16 to 18 around iterations 58 to 70, and ended at about 24. On the evaluation posts PPO summaries are 22.5 tokens on average, SFT 30.72, DPO 44.32.

A higher reward-model score is not a better summary. On the 300 evaluation posts PPO has the highest mean reward (4.75 vs 4.30 for SFT) and wins against SFT on 58.7% of posts, but ROUGE-L drops from 0.254 to 0.193. In my blind human evaluation of 30 posts the picks were SFT 14, PPO 1, DPO 15. PPO had the highest reward and was picked once.

The reason is reward hacking. PPO learned to end many summaries with an emotion phrase that the reward model likes. Counts over the 300 evaluation summaries (SFT / PPO / DPO): "incredibl" 0 / 24 / 2, "disappoint" 0 / 63 / 0, "much " 0 / 10 / 3. Examples of PPO summaries:
- "Attempting to have sex with new girlfriend and feeling uncomfortable, but still disappointed."
- "GF and I have been having problems regarding sex and anxiety, we are disappointed."
- "My ex boyfriend and I broke up for 8 months and now we are planning to break up. Incredibly lonely."
- "Short flight, no response from company, and no response to interview, feeling disappointed."
- "Attempted relationship with new girl and I feel disappointment and disappointment."

Some of these also contain wrong facts. The reward model is only 61% accurate, so it is a weak proxy, and PPO found a pattern it overrates. The opposite also happens: DPO has the lowest reward (3.49) but was picked 15 times, so the reward score does not track my preference in either direction. Caveats: one rater, 30 posts, so SFT 14 vs DPO 15 is not a real difference; PPO 1 of 30 is a clear result. Examples are in results/samples.md.

## 7.3 Role of beta_KL

In the KL-constrained objective (Eq. 41) the policy maximises the reward-model score minus beta_KL times the KL to the reference. In my implementation this is a penalty of -beta_KL (log pi_old - log pi_ref) on every token, and the reward-model score is added only at the last token. beta_KL is what keeps the policy close to SFT, so it keeps the text fluent and stops the policy from moving to places where the reward model has not seen data.

With beta_KL = 0 the only signal is the reward-model score. Nothing would hold the policy near SFT, so I would expect the KL to grow faster and without a limit, and the policy to exploit the reward model even more: more of the emotion-phrase endings, repeated words, and summaries that score high but are not readable, with ROUGE-L and human preference falling further. My run already shows this at beta_KL = 0.05 (KL about 30 and the "disappointed" pattern), so a weaker or zero penalty would be worse. I did not run this.

## 7.4 Clip fraction

The logged clip fraction is the average over the 4 PPO epochs of the share of tokens with ratio outside [0.8, 1.2]. In epoch 1 it is exactly 0 in all 200 iterations (the ratio is 1), so the logged value is pulled down by that epoch. It was 0.15 at iteration 1, about 0.03 around iteration 25, 0.15 to 0.23 around iterations 45 to 90, and about 0.05 to 0.07 at the end. Averaged over blocks of 10 iterations it goes 0.05, 0.04, 0.12, 0.16, 0.15, 0.14, 0.09, 0.06, 0.06, 0.05.

When the clip fraction is high, many tokens reach the clip limit in epochs 2 to 4, so each update is large and the clipping is actually limiting it. This is the same period where the KL rose fastest and the summaries became shortest (iterations about 45 to 90). Later the clip fraction is small, so the policy changes less per iteration and the KL and length settle.

## 7.5 DPO: where the reward went

The KL-constrained objective has a closed-form optimal policy: pi*(y|x) = pi_ref(y|x) exp(r(x,y)/beta) / Z(x). Taking logs and rearranging gives r(x,y) = beta log(pi*(y|x)/pi_ref(y|x)) + beta log Z(x). So the reward can be written with the policy and the reference. DPO uses beta log(pi_theta(y|x)/pi_ref(y|x)) as an implicit reward, so no reward model is trained.

Z(x) is a sum over all possible summaries, which is impossible to compute. It does not depend on y. In the Bradley-Terry loss only the difference r(x,y_w) - r(x,y_l) appears, and both terms contain the same beta log Z(x), so it cancels. What is left is the DPO loss with beta = 0.1 that I implemented.

Accuracy on the same 1,000 validation pairs: DPO implicit reward 0.608, reward model 0.614. The difference is 6 pairs out of 1,000, far inside the noise (standard error about 1.5 points), so they are about equal. DPO reached this without a separate reward model. Important: DPO ran only 1,073 of the 2,500 steps (about 43% of one epoch), because it hit my 70 minute budget (3.75 s/step). The 250-step checks used only the first 400 validation pairs (0.6325, 0.59, 0.6025, 0.5975), so they are not directly comparable to the final 1,000 pair number.

DPO also behaves differently on generation. It stayed much closer to SFT (KL to SFT 2.82 vs 11.98 for PPO), but 183 of the 300 DPO summaries reached the 48 token limit without ending (SFT 31, PPO 1), which gives the mean length of 44.32. A summary that is cut off is probably also out of distribution for the reward model, which would be one reason for its lower reward (3.49, win rate vs SFT 0.4533). I did not test this. With a full epoch the result could be different.

## 7.6 Models in memory, time, and which method

PPO needs three full models in GPU memory: the trainable policy (with a small linear value head on its last hidden layer), the frozen reference, and the frozen reward model. DPO needs two: the trainable policy and the frozen reference. In float32 one 135M model is about 0.54 GB of weights; the trainable policy also needs gradients and the two AdamW states, about four times that. On the T4 the activations were the larger problem, which is why batches were split into halves or chunks.

Time per stage on a T4: SFT 75.87 min, reward model 116.92 min, PPO 48.98 min (200 iterations), DPO 72.08 min (1,073 of 2,500 steps; at 3.75 s/step a full epoch would be about 156 min). PPO needs the SFT model and the reward model first, so the PPO route cost 75.87 + 116.92 + 48.98 = 241.77 min. The DPO route needs no reward model: 75.87 + 72.08 = 147.95 min.

For a new project I would start with DPO: two models instead of three, no rollouts, no reward model stage, and in my runs it stayed close to SFT and showed no reward hacking. I would use PPO only if I had a reward model I trust and could check outputs by hand while it trains, because mine was exploited quickly. I cannot claim DPO is better than SFT from my data: it was trained for 43% of an epoch, its summaries were often cut off, and human picks were SFT 14 vs DPO 15 from a single rater.
