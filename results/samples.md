# Qualitative samples (10 evaluation posts)

Greedy decoding, at most 48 new tokens. Numbers in brackets are the reward-model score. The summaries are copied unchanged from results/eval_summaries.jsonl. "Post end" shows the last 250 characters of the post.

## Sample 1 (eval post 195): PPO clearly better than SFT

**Post end:** this now. She's the only girl I've been with, that's the only thing I can think of that drove my stupid brain to do such a dumb thing. I want to erase everything that's happened the last 48 hours! God damn. Sorry for blurting at the end there. TL;DR:

**Reference:** ? Please read some at least. I need some help here. I'm in a bad way, feel terrible. Sort of hate myself.

- **SFT** (-0.59): I'm a girl, I'm a girl, I'm a girl, I'm a girl, I'm a girl, I'm a girl, I'm a girl, I'm a girl, I'm a girl, I'm a
- **PPO** (4.66): I went to website and paid money for sex and then followed up with girl. Incredible stupid mistake.
- **DPO** (0.38): I joined a website that was basically for hooking up, paid a couple of days trial of full service, got girlfriend of over a year, I want to erase everything that's happened the last 48 hours! God damn.

**Why:** SFT degenerates into a loop ("I'm a girl") and says nothing about the post. PPO states the event (paid a website for sex, regrets it). DPO is on topic but copies the last sentence of the post and is cut off at 48 tokens.

## Sample 2 (eval post 253): PPO clearly better than SFT

**Post end:** ur body is just not maintainable for a life style in my opinion. I can't remember the exact quote, but Layne Norton said something like " If you can't see yourself on the staying on the diet 10 months from now, than you shouldn't be doing it." TL;DR:

**Reference:** I think low carb is not the best option for most, I think flexible dieting is. I have experienced an eating disorder from low carb, and successfully lost all the weight I gained from binge eating.

- **SFT** (-0.44): I'm not sure why low carb and four hour body are suggested in the FAQ. I'm not sure if it's because it's a diet/binge cycle or if it's because it's a diet/binge cycle that I
- **PPO** (4.73): Low carb diet proposals are ridiculous and irresponsible. People will fall into eating disorders because of low carb diet.
- **DPO** (2.70): Why is low carb suggested in the FAQ? Why is it a rigid diet that leads to eating disorders in the long run? I've tried low carb and four hour body and found it to be a mental drag to stay on the low carb

**Why:** SFT repeats the same clause until the length limit. PPO gives the author's actual opinion (low carb leads to eating disorders) in two short sentences, a bit exaggerated. DPO is on topic but runs to the 48 token limit.

## Sample 3 (eval post 83): DPO clearly better than SFT

**Post end:**  I'm going for the food because I don't dance."  I am constantly getting bombarded by my mother about the catering and she is concerned with nothing else. I feel as if she has lost sight of her family and is stressing me out about the wedding. TL;DR:

**Reference:** Advice needed on fixing the relationship between me [23F] and my mom [51F] who acts like a toddler.

- **SFT** (0.91): My mom is a spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled,
- **PPO** (4.03): My mother has become excessively needy and intrusive and I feel disappointed.
- **DPO** (6.39): My mom [51] and I used to be a fantastic relationship. She now focuses on herself and acts like a toddler. I feel as if she has lost sight of her family and is stressing me out about the wedding.

**Why:** SFT loops on the word "spoiled". DPO names the mother, the toddler-like behaviour and the wedding stress (mostly reusing the post's own sentence). PPO is short and vague and ends with "I feel disappointed".

## Sample 4 (eval post 189): PPO clearly worse than SFT

**Post end:** look and the other people at the bus stop sort of look at me oddly. She then turns to someone else and asks "will you tell me when my bus is coming?"  I'm standing there while the guy says yes and I feel and look like the world's biggest knob. TL;DR:

**Reference:** I misheard a disabled lady asking me to tell her when her bus is coming because she can't see very well and proceeded to reject her like a douche.

- **SFT** (4.32): I was standing at the bus stop and a disabled lady said "Is the bus coming?" and I replied "No, I don't think so" and she gave me a sort of hurt look.
- **PPO** (2.39): Attempting to be polite and rude and fail to realize that she is mentally disabled. Attempting to be rude and rude.
- **DPO** (4.57): I was standing at the bus stop waiting for a bus with my earphones in and was hearing a woman talking to one or two people there. She turned to me and said "will you tell me when my bus is coming?"

**Why:** SFT tells the story correctly. PPO misses it (she asked when her bus is coming), repeats "rude and rude" and has the lowest ROUGE-L. DPO sets up the scene but copies the post and has no point.

## Sample 5 (eval post 37): PPO clearly worse than SFT

**Post end:** dmittedly I'm a little jealous and wish I could work the system in such a way.  Still I am hesitant to fully point the finger on the slight chance that it really is as bad as his medical history claims but I can't help but be skeptical.  /rant TL;DR:

**Reference:** former supervisor has plethora of mental illness and is repressed homosexual. as a result barely works but maintains 100% entitlements. I'm sort of skeptical and jealous.

- **SFT** (2.82): old supervisor of mine has anxiety/depression and is also a closet homosexual. Needless to say he goes to see medical professionalsals a lot.  I'm not going to say he's malingering on purpose but he only works
- **PPO** (1.88): Misfortune/depression/homosexuality causes intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive intrusive
- **DPO** (2.82): old supervisor of mine has anxiety/depression and is also a closet homosexual. Needless to say he goes to see medical professionalsals a lot.  I'm not going to say he's malingering on purpose but he only works

**Why:** PPO collapses into repeating "intrusive" until the 48 token limit (reward 1.88 vs 2.82 for SFT). DPO output is identical to SFT here.

## Sample 6 (eval post 79): DPO clearly worse than SFT

**Post end:** ely I have even found myself acting more demure and feminine, even at work.  So I feel like it's spilling over into other parts of my life.  It's hard to assert manliness throughout the day when you're manhood is tucked into a pair of panties. TL;DR:

**Reference:** Gf's fetish for femininity is making me a little uncomfortable and even altering how I act and think throughout the day.

- **SFT** (5.76): Gf is getting off on me to dress her in lingerie and put makeup on me.  Is this normal?  Should I be more masculine or feminine?
- **PPO** (5.22): Gf has been dressing me in lingerie and I feel uncomfortable and disappointed.
- **DPO** (-0.57): Gf [23f] "gets off" on having me do feminine things.  She likes me to wear my panties under my regular clothes all day.  But now she even likes me to wear her panties under my

**Why:** SFT is short and on topic. DPO copies long pieces of the post, runs to 48 tokens and stops mid-sentence (reward -0.57). PPO is fine here, and it is the one post where my human pick was PPO (post 11 of the blind sheet).

## Sample 7 (eval post 0): Mixed, SFT best

**Post end:** e ... also I'm big on comuncating my feelings , but I'm petrified telling him his words hurt me, will result in another fight like yesterday's one. What should I do ? Because at this point I really just want to be done with him and move on ... TL;DR:

**Reference:** Boyfriend said some would stuff to me and I am so deeply hurt by them, but I'm scared bringing it up will result in another argument in which he ruins me once more.

- **SFT** (5.95): boyfriend said hurtful things to me and I'm still deeply hurt by it. What should I do?
- **PPO** (5.69): My boyfriend hurt my feelings and made horrible comments to make me feel horrible and hurt. We are still in shock.
- **DPO** (2.53): boyfriend said incredibly hurtful things to me and I'm petrified to tell him his words hurt me. What should I do ? I want to hurt the way he hurt me and move on but I'm petrified to tell him

**Why:** SFT matches the reference closely (ROUGE-L 0.38). PPO turns it into vague emotion words ("horrible", "in shock"). DPO copies the post and is cut off at 48 tokens.

## Sample 8 (eval post 135): Mixed

**Post end:** lled my lady and said hey you gotta clear your plate, he is a long time friend and he needs me there for support. She said no, not going, then posted on Facebook I was just leaving her alone on NYE. What can I do about this? Am I being unfair? TL;DR:

**Reference:** Do I go to my friends on NYE or with her? If I comply with her does she hold all the power

- **SFT** (0.17): I am not ready for a baby, I am not ready for a relationship, I am not ready for a relationship, I am not ready for a relationship, I am not ready for a relationship, I am not ready for a relationship,
- **PPO** (4.86): My boyfriend and I have been together for 6 months and we have been fighting. feeling disappointed.
- **DPO** (5.67): Girlfriend broke up with me, I said I was not ready, she lost her shit, now I think she feels she can control me. What can I do about this?

**Why:** SFT loops about a baby, which is off topic. DPO is on topic (being controlled) but invents a breakup. PPO invents a boyfriend and ends with "feeling disappointed", a generic emotion ending.

## Sample 9 (eval post 73): Mixed

**Post end:** an awesome present she won't get to see on valentines day because I am forbidden to see her at all. Not to mention I got my phone taken away so now I have no way to talk to her at all! I feel like I disappointed her and it's killing me inside. TL;DR:

**Reference:** Can't see or talk to my girlfriend on valentines day and I feel depressed. I don't know what to do.

- **SFT** (4.96): Girlfriend got banned from school for a month and I got banned from school for a month. I'm missing a lot of things because of this and I don't know when it's going to stop.
- **PPO** (5.10): My boyfriend tried to sneak over to my girl's house and I feel terrible because of it.
- **DPO** (-1.50): I got caught trying to sneak over to my girlfriend's house when her parents weren't home and I got caught and my girlfriend got banned from seeing me at all. I'm missing a lot of things because of this and I feel like I

**Why:** SFT invents a school ban. PPO swaps the people ("my girl's house") and ends on "I feel terrible". DPO is closest to the post but copies it, stops mid-sentence and gets the lowest reward (-1.50).

## Sample 10 (eval post 210): All three poor

**Post end:** n the closest friends and I don't believe I've said anything that could have upset her. In fact, I haven't really hung out with her in a long while. Any ideas of why she might not want to see me, Reddit? Any advice on how to get closer to her? TL;DR:

**Reference:** A girl has some unknown "shit" with me and doesn't want to see me when I haven't done anything that might have upset her.

- **SFT** (4.39): I'm going to see Wicked tonight and I want to know why she had shit with me.
- **PPO** (2.71): My friend and I went to see Wicked theater tonight and we all went to see it. We all seemed to be happy. We all seemed to be friends. Was it shit?
- **DPO** (2.13): I was all set to go to see Wicked (a Broadway play) tonight and I was all set to hang out with a girl in the group. She told me that one of the girls in the group had shit with me.

**Why:** All three focus on "Wicked" and "shit" and none asks why the girl does not want to see the author. PPO adds a repeated "We all seemed..." pattern. No clear winner.
