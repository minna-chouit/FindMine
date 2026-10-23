# FindMine user test (blindfolded volunteers)

**Question:** blindfolded, do people find *their* item faster and pick the wrong one less often with FindMine (B)
than with a "what kind of object" app (A, like current apps)? About **10 minutes per person**.

## Setup

- Two registered items (`item1`, `item2`, e.g. juice bottle and black backpack), each with its **look-alike on the
  table in every round**, plus normal clutter. A lamp for the low-light rounds.
- Phone with the app (volume up, screen lock off), blindfold, a second phone as stopwatch.
- Two people: the **facilitator** (times, fills `eval/user_trials.csv`) and the **spotter** (stays next to the
  volunteer, no hints).
- Mode A = Settings → Test mode → **Baseline (test A)**. Mode B = **Full app**. Check the mode before each round.

## Per person

1. Short explanation + the Google Form's consent question (30 s). Say: "You can stop at any time."
2. One practice search without the blindfold.
3. **4 rounds**, in the order of the person's rows in `eval/user_trials.csv` (person 1 = first volunteer, …; the
   order alternates so A and B each go first half the time). Move the items to a new spot between rounds.
   - Say **"Find your [item]. Go."** and start the clock.
   - **Stop** when they hold an object and say "got it", or at **2:00**.
   - Result: `found` (their item), `wrong_item` (the look-alike or something else), `timeout`, `gave_up`.
   - Write the time and the session id shown in the app.
4. Blindfold off → they fill in the Google Form (1 minute).

## After everyone

Download the form's responses as CSV to `eval/form_responses.csv`, then:

```
.venv/Scripts/python ml/summarize_user_tests.py
```

It writes `eval/user_tests_summary.md`: success rate, median time and wrong-item picks per mode × light, the mean
of each 1–5 question and the written answers. Names never appear in the summary. Stop the EC2 instance afterwards.

## Safety

The spotter is always within arm's reach; clear the floor; nobody is filmed without saying yes in the form.
