# FindMine accessibility check

Users: **blind** people (screen reader or no screen at all) and **low-vision** people (large text, high contrast, magnification). Target: WCAG 2.2 level AA everywhere, AAA where we can.

## Verdict

| Who | Search (find my item) | Registration | Overall |
|---|---|---|---|
| **Low vision** | ✅ Good | ✅ Good | **Good**: meets AA, colours and text at AAA |
| **Blind** | ✅ Good: voice in, speech + beeps + vibration out | ✅ Mostly: name by voice, photos taken automatically, spoken guidance (the type list still needs a tap or a helper) | **Good** |

Registration can be done independently by voice or with a sighted helper (requirement F1.1). Photos are taken automatically when the server's OpenCV camera coach says the picture is sharp, lit and steady, with rising beeps while it gets ready.

## What passes

| Check | Rule | Result |
|---|---|---|
| Text contrast | WCAG 1.4.6 (AAA, 7:1) | ✅ Every pair 7:1 or more, light and dark (table below) |
| Control borders | WCAG 1.4.11 (3:1) | ✅ 5.4:1 to 7.3:1 |
| Text size | 1.4.4 resize to 200 % | ✅ All sizes in `rem`; pinch-zoom allowed; in-app **Large** and **Extra large** |
| Font | — | ✅ Atkinson Hyperlegible, 18 px base, bold buttons |
| Tap targets | 2.5.8 (AA 24 px), 2.5.5 (AAA 44 px) | ✅ 52–126 px; main actions 84 px and more |
| Not colour only | 1.4.1 | ✅ States have words (chip, Yes/No, "match 82 %"), not only colours |
| Keyboard / switch focus | 2.4.7, 2.4.13 | ✅ 4 px indigo outline + butter halo on every control |
| Labels | 1.3.1, 4.1.2 | ✅ Every input has a label; icon-only buttons have `aria-label`; emoji hidden from screen readers |
| Live updates | 4.1.3 | ✅ Status and answers in `aria-live` regions |
| Language | 3.1.1 | ✅ `lang` and right-to-left `dir` switch for Arabic |
| Non-visual output | project rule "nothing only visual" | ✅ Speech, stereo beeps, vibration; boxes on the video are extra |
| Safety | project rule | ✅ Spoken notice: not a navigation or mobility aid; "found" only at ≥ 0.8 or after "yes" |
| User preferences | — | ✅ Follows system dark mode, high-contrast mode, Windows forced colours, reduced motion |

## Contrast table

| Pair | Light | Dark |
|---|---|---|
| Text on page | 14.1:1 | 15.9:1 |
| Secondary text on page | 7.8:1 | 11.3:1 |
| Text on butter button | 11.4:1 | 10.9:1 |
| Text on periwinkle button | 11.6:1 | 8.5:1 |
| Stop / No button | 8.2:1 | 8.4:1 |
| Yes / Got it button | 7.0:1 | 9.8:1 |
| Text over the waves | 7.0:1 or more (secondary), 12.2:1 (text) | 7.9:1 or more |

## Still to do (honest list)

| Gap | Impact | Fix |
|---|---|---|
| Object type is a list | Needs one tap (or a helper) | Could be guessed from the detector later |
| Not yet tested with **TalkBack** | Unknown double-speech (TalkBack + app voice) | Day 10 accessibility pass with TalkBack on |
| Phone may lack an **Arabic voice** | Silent in Arabic | Settings shows a warning + how to install it |
| **iPhone**: no vibration from the web | Less feedback on iOS | Beeps and speech still work |
| Font loads from Google Fonts | Falls back to system font offline | Fine; could self-host later |

## How to test quickly

1. Android → Settings → Accessibility → **TalkBack** on. Open the home page and do a whole search without looking.
2. Settings → Display → **Font size** largest + **Display size** largest: nothing should be cut off.
3. In FindMine Settings, try **Extra large** text and **Dark**.
4. Chrome DevTools → Lighthouse → Accessibility: aim for 100.
