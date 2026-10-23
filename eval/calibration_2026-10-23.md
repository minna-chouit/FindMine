# Verifier calibration on FindMine-Home (Oct 23)

Data: 10 personal items in 6 classes photographed at school (bags ×3, bottles ×2, keys ×2, cup, phone, headphones).
Cables (adapter, charger, wired earphones) were skipped: the 12-class detector has no cable class.

What we found:
- The proxy weights (fitted on synthetic COCO views) scored real items far too low: at *found* = 0.8, recall was
  0.05; the live bottle search (session 0b36fbe0) stayed at 0.19–0.27 and never asked.
- Reference photos are close-ups with background; live crops are tight boxes. Reference photos are now cropped to
  the detected item (`crop_to_item`) in both the evaluation and the server's registration/loading.
- The `test/with_other` shots did not keep the left/right rule (both bottles right of centre; the same video in two
  folders), so they were left out (`--no-with-other`). They need reshooting.

Calibration (`ml/eval_home.py --no-with-other --fit`): 81 positive / 69 look-alike rows (alone frames + each
reference photo against the item's other references and the look-alike's). cv.ml logistic regression:
bias −1.81, embedding 1.73, geometry 0.57, colour 1.02. Mean cues own vs look-alike: embedding 0.80 vs 0.68,
geometry 0.11 vs 0.06, colour 0.57 vs 0.45.

Thresholds: the precision rule gave *found* 0.55 (precision 0.96, 2 of 75 look-alikes above it). Because a wrong
"found" is the worst error, we raised *found* by hand to **0.62**: precision 1.00 on these rows, 0 of 75
look-alikes above it (the highest, 0.593, is one black bag against the other). *ask* = **0.48** (precision 0.70):
below *found* the app only asks "is it yours?", which is safe.

Limits: in-sample and small (two black bags decide the *found* threshold); far, small items (bottles at 2–3 m)
mostly fall in the *ask* band; keys and headphones are not detected in room photos (detector recall).
