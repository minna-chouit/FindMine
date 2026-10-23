# FindMine user test protocol (blindfolded volunteers)

Plan reference: Level 3 "Human outcomes" and requirement N4 (success within 2 min ≥ 80% in normal light, ≥ 60% in low light/clutter).

**Question we answer:** with a blindfold on, do people find *their own* item faster and with fewer wrong pickups using
FindMine (B) than with a "what kind of object is it" app (A, the way current apps work)?

About **20–25 minutes per person**. Aim for **6–8 volunteers**.

## Roles (2 people minimum)

- **Facilitator:** reads the scripts, starts/stops the timer, fills `eval/user_trials.csv`.
- **Spotter:** stays next to the volunteer the whole time, keeps them away from edges, stairs, hot or sharp things.
  The spotter never gives hints.

## Before the day

- [ ] Server deployed and `https://<server>/health` answers; phone charged, volume up, screen lock off.
- [ ] Items registered in the app (5 photos each): **2 target items** (e.g. keys, mug) and for each one a **look-alike**
      of the same class (other keys, other mug). Write down each item's `item_id`.
- [ ] Practice item registered (e.g. a bottle) for the warm-up.
- [ ] Room: a table + a shelf or sofa, normal clutter (books, cables, a bag). A lamp for the low-light condition.
- [ ] Blindfold (sleep mask), stopwatch (phone of the facilitator), printed consent forms (`eval/consent_form.md`).
- [ ] Optional video: a second phone on a stand, filming hands and table only (no faces).

## Conditions

| | System | What the phone does |
|---|---|---|
| **A** | Baseline (`"mode": "baseline"`) | Announces the first object of the right *class* ("Found your keys at 2 o'clock"), mine or not. |
| **B** | FindMine | Coach + detector + "is it mine?" check + asks before "found" + guidance. |

| | Light | Room |
|---|---|---|
| **normal** | Room lights on | Item on the table among 4–5 other objects |
| **low_clutter** | Main light off, one lamp | Item partly hidden among 8–10 objects (half under a cloth, behind a cup…) |

Each volunteer does **8 trials**: 2 systems × 2 lights × 2 items. A look-alike is in the room in half of the
trials (column `lookalike` in the sheet). The order is already written in `eval/user_trials.csv` and alternates
between volunteers (P1 starts with A, P2 with B, …) so learning effects cancel out. **Follow the sheet's order.**

## Script

### 1. Welcome and consent (3 min)

> **EN:** "Thank you for helping. FindMine is a phone app that helps blind people find their own belongings.
> You will wear a blindfold and use the phone to find objects on a table. It is the app we are testing, not you.
> [Name of spotter] stays next to you the whole time. You can stop at any moment, for any reason.
> We record times and, if you agree, a video of your hands only. Is that OK?"
>
> **FR :** « Merci de nous aider. FindMine est une application qui aide les personnes aveugles à retrouver leurs
> propres affaires. Vous allez porter un bandeau et utiliser le téléphone pour trouver des objets sur une table.
> C'est l'application qu'on teste, pas vous. [Nom] reste à côté de vous tout le temps. Vous pouvez arrêter à tout
> moment, sans raison. On note les temps et, si vous êtes d'accord, une vidéo de vos mains seulement. D'accord ? »

Get the consent form signed. Note age range and whether they wear glasses (column `notes`).

### 2. Practice (3 min, not timed)

Without blindfold, show how to hold the phone at chest height and sweep slowly. Then blindfolded, one practice
search for the practice item with system B: let them hear "Is it yours?", press **Yes**, follow the beeps, press **Got it**.
Answer questions now: no help once the trials start.

### 3. Trials (8 × max 2 min)

Between trials the volunteer turns away (or the spotter leads them a step back) while the facilitator places the
item (and the look-alike if the sheet says so) in a **new spot** each time.

For each trial:

1. Facilitator sets the system (A/B) and light, chooses the item in the app, notes the `session` id shown by the app.
2. Says: **"Find your [item]. Go."** and starts the stopwatch.
3. **Stop** the clock when the volunteer **holds an object and says "got it"** (or presses Got it), or at **2:00**.
4. Write the result:
   - `found`: holds **their** target item.
   - `wrong_item`: holds the look-alike or another object and says got it. (Gently tell them, then continue.)
   - `timeout`: 2:00 reached.
   - `gave_up`: they stopped.
5. Count `coach_prompts` (how many times the app said "too dark", "slow down"… : or read it from `/trace/<session>`).

No hints from anyone during a trial. If the app crashes: note `app_error` in `notes`, restart, repeat the trial once.

### 4. Questionnaire (5 min, blindfold off)

The 10 SUS statements about **FindMine (B)**, answered 1 (strongly disagree) to 5 (strongly agree), then the two open
questions. Write answers in `eval/sus.csv`.

| # | English | Français |
|---|---|---|
| 1 | I think that I would like to use this system frequently. | J'aimerais utiliser ce système souvent. |
| 2 | I found the system unnecessarily complex. | J'ai trouvé ce système inutilement compliqué. |
| 3 | I thought the system was easy to use. | J'ai trouvé ce système facile à utiliser. |
| 4 | I think that I would need the support of a technical person to be able to use this system. | J'aurais besoin de l'aide d'une personne technique pour utiliser ce système. |
| 5 | I found the various functions in this system were well integrated. | Les différentes fonctions du système sont bien intégrées. |
| 6 | I thought there was too much inconsistency in this system. | Il y a trop d'incohérences dans ce système. |
| 7 | I would imagine that most people would learn to use this system very quickly. | La plupart des gens apprendraient à utiliser ce système très vite. |
| 8 | I found the system very cumbersome to use. | J'ai trouvé ce système très lourd à utiliser. |
| 9 | I felt very confident using the system. | Je me suis senti(e) très en confiance en utilisant ce système. |
| 10 | I needed to learn a lot of things before I could get going with this system. | J'ai dû apprendre beaucoup de choses avant de pouvoir utiliser ce système. |

SUS: Brooke (1996), free to use with attribution. The French wording is ours, not the validated French SUS.

Open questions:
1. "What was the most helpful thing the app said or did?" / « Qu'est-ce qui vous a le plus aidé ? »
2. "What was the most confusing or annoying moment?" / « Quel moment était le plus confus ou agaçant ? »

Write down one or two exact quotes: they go in the report and the video (with permission).

### 5. After everyone

```
.venv/Scripts/python ml/summarize_user_tests.py
```

Prints and writes `eval/user_tests_summary.md`: success rate, median time-to-find and wrong-item pickups per
system × light, and the SUS score (0–100; 68 is average). Then **stop the EC2 instance**.

## Ethics checklist

- Signed consent before anything else; the right to stop at any time, said out loud.
- Spotter always within arm's reach of a blindfolded volunteer; clear the floor of obstacles.
- No faces in videos or photos unless the person agreed in writing; live frames are not stored by the server.
- Names never go in the CSVs: use P1, P2, ….
