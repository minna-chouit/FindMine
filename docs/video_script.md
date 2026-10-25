# FindMine video: shot list and script (target 4:45, hard limit 5:00)

The rules ask for: **the team, the application working, its architecture, and its principal results.**
Film vertical clips for social media separately; this video is **horizontal (16:9)**. Captions on everything
(our users and judges may watch without sound). Never show the `.env` file, AWS keys, emails or passwords.

| Time | Shot | What we show | Voice-over / on screen |
|---|---|---|---|
| 0:00–0:20 | Close-up: a hand sweeping a cluttered table, eyes covered | the problem | "For a blind person, a lost cup means searching every surface by hand. Apps can tell you *a* cup is there. Not *your* cup." |
| 0:20–0:35 | Both of us at the desk, then names on screen | **the team** | "We're [names], and this is FindMine." |
| 0:35–1:05 | Registration: phone takes 5 photos with beeps, voice naming | register once | "Register an item once: five photos, guided by voice, taken automatically when the picture is sharp." |
| 1:05–2:20 | **The hero clip** (blindfolded teammate, look-alike cup on the table), two angles: person + phone screen recording with sound | **the app working** | Let the app speak: "Something like your cup at 2 o'clock. Move closer so I can check it's yours." → "Is it yours?" → "Yes" → beeps → hand on the cup → "Got it". Caption the look-alike: "same kind of cup, not hers". |
| 2:20–2:40 | Dim room: coach says "too dark", torch switches on | the agent acting on what it sees | "When the picture is bad, FindMine says how to fix it, and the agent can turn on the torch." |
| 2:40–3:20 | Architecture diagram, then the `/trace` page of the hero search | **architecture** | "The phone streams frames to OpenCV 5 on AWS: a camera coach, a YOLO detector in cv.dnn, and a check that compares the object with your photos. A state machine decides every frame; Claude on Bedrock plans only when the search is stuck. Every decision is logged." |
| 3:20–4:15 | 3–4 result cards (big numbers, one per card) | **principal results** | ① user test: found within 2 min, A vs B [TODO] · ② 0 look-alikes ever called "found" (0/75 same kind, 0/904 other kinds) · ③ blur hint: item found in 45% of frames instead of 24% · ④ "yours vs look-alike": AUC 0.91 up close vs 0.37 from afar, so FindMine moves you closer before asking |
| 4:15–4:35 | One failure, honestly: keys | limitations | "Keys are hard: small and shiny. [TODO run 2 result]." |
| 4:35–4:45 | Both of us, the link on screen | close | "FindMine. Built with OpenCV 5 on AWS. Try it at [link]." |

## Clips we need (matches the Cadence shoot tasks)

- [ ] Hero search, 2 angles (phone screen recording **with audio** + second phone filming the person) · 2–3 takes
- [ ] Registration screen recording with beeps
- [ ] Dim room + torch clip
- [ ] `/trace/<session>` scrolled for the hero search (screen recording)
- [ ] Team intro (horizontal) · desk B-roll
- [ ] Architecture diagram (export from the report)

## Editing checklist

- [ ] Under 5:00 (aim 4:45) · captions · no secrets on screen · volunteer faces only with written consent
- [ ] Upload as **Unlisted** on YouTube · check the link in a private window · put it in the report and the submission
