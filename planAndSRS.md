# FindMine — OpenCV AI Competition 2026 Battle Plan & SRS

Oct 15, 2026 · @mina

## TL;DR — what we build and why it wins

**FindMine** is a phone web app that helps blind and low-vision people find **their own** belongings — *my* keys, not just “a key”. The user teaches it an item with a few photos, says “find my keys” (Arabic, French or English), and sweeps the phone. An **active-perception agent** on AWS watches the camera through OpenCV 5, coaches the user to fix bad camera shots, remembers which directions were already scanned, asks before it says “found” when unsure, and guides the hand in with clock-face speech, stereo beeps and vibration.

**The gap we exploit.** Your competitor list shows that everyone does *category* detection plus a description (Seeing AI, Lookout, Be My AI…). Almost nobody does three things well, and those three are our pillars:

1. **Instance recognition — “mine, not any.”** Few-shot matching of a specific item from 3–5 reference photos, checked with classic OpenCV geometry (ORB/SIFT + RANSAC homography) on top of a learned embedding. Judge Gary Bradski co-authored ORB; using it well, and measuring it, lands with this panel.
2. **Camera coach.** Blind users take blurry, dark, badly framed or lens-blocked shots — it is the #1 failure in the VizWiz research. We detect blur, exposure, shake and lens blockage with pure OpenCV, give one concrete fix (“turn on the torch”, “slow down”, “wipe the lens”), and **check that the fix worked**.
3. **An agent that searches actively.** Vision output decides the next step: rescan the area you skipped, switch on the torch, zoom/tile a region, ask “there are two black wallets — the one on the left?”, or fall back to “last seen in the kitchen, 14:02”. That is exactly the Agentic Vision definition.

**Prize targets (all three can stack):**

| Prize | Value | How we qualify |
| --- | --- | --- |
| Grand prize / top 3 | $5,000 / $3,000 / $2,000 | Deep OpenCV 5 use, real users, measured results, polished demo |
| Agentic Vision Award | $1,000 | Perceive → decide → act loop; logged traces show OpenCV output changing the next tool call; human confirms uncertain matches |
| Best Use of COOL Award | $1,000 | Vision server runs on Graviton with COOL; reproducible benchmark against stock OpenCV on x86 and on Arm |

**How the plan maps to the judging weights:**

| Criterion | Weight | Our evidence |
| --- | --- | --- |
| Technical execution (OpenCV 5) | 30% | DNN detector + embedding via `cv.dnn`, ORB/SIFT + RANSAC, tracker, Kalman, optical flow, quality metrics, barcode/QR; ablation tables |
| Innovation | 20% | Personal-instance search + camera coach + coverage memory, combined in one agent loop |
| Real-world impact | 20% | Tests with blindfolded volunteers and, if possible, 2–3 visually impaired users; time-to-find and success-rate numbers; Arabic/French support for North Africa |
| User experience | 10% | Voice-first, one-button, screen-reader compatible, audio + haptic guidance |
| Documentation | 10% | Report, diagram, 5-min video, one-command setup |
| Cloud, reproducibility, responsible use | 10% | IaC deploy, CloudWatch dashboard, privacy defaults, face blurring in logs, failure-case gallery |

**Rule of the next 11 days:** *one flow, done excellently, measured honestly.* Everything else on your competitor list (currency, documents, stairs, obstacles) is out of scope unless the core is finished on Day 7.

## Assumptions to confirm today (Day 1, 30 minutes)

The plan assumes the points below. If one is wrong, change it now. Changing it on Day 6 costs days.

- **Team (confirmed):** 2 people, **no fixed leads**; both can do every kind of task. The schedule uses two **tracks**: **A = CV/ML** (data, training, OpenCV pipeline, evaluation) and **B = App/Cloud/Agent** (web app, AWS, Bedrock agent, deployment). They are not people. Each morning, each person claims tasks in the tracker app. Rule of thumb: the same person keeps a track for the whole day, so you don't switch context. Both write the report and the video.
- **Client (confirmed):** a **mobile web app (PWA)** in Chrome on Android. Camera, microphone, speech, vibration and the torch all work from the browser, and judges open a URL with no install. iOS Safari works, but without vibration or torch.
- **Compute (open):** the rules require a meaningful part of the system to run **on AWS**. The vision service runs on a Graviton server (`c7g.large`) in `eu-west-3` (Paris) or `eu-south-1` (Milan). The phone streams downscaled frames (\~5 fps, 640 px) over a WebSocket.
- **Training GPU:** use your **$200 NVIDIA platform credit** (e.g. a single L4/A10/T4-class GPU instance, a few hours per run), with free Colab/Kaggle as backup. A YOLO-nano/small fine-tune takes about 2–4 h on such a GPU, so $200 is far more than you need. NVIDIA credits **cannot replace AWS**: the deployed system must run on AWS.
- **AWS money (confirmed: $0 budget):** an AWS Free plan account ($100 credits + up to $100 more for completing activities; no billing on the Free plan). Estimated use: ≈ $30–50 (see the zero-budget plan in the risks section). **You do not need more money**, as long as the Free plan allows the services you need.
- **OpenCV version:** OpenCV **5.x** everywhere, pinned to one exact version. How to pin it: run `pip index versions opencv-contrib-python` to see what exists; if a 5.x version is listed, put `opencv-contrib-python==5.<x>.<y>` in `requirements.txt` and install with `pip install -r requirements.txt`. Both laptops and the Docker image use that file. At server start, log `cv2.__version__`. If no 5.x wheel is on PyPI yet, use the official OpenCV 5 build route on Day 1 and write it down. Every vision step goes through OpenCV 5 APIs, not PyTorch, at inference time.
- **LLM (being checked):** Amazon Bedrock (Claude, a small model such as Haiku first) for the agent's planning. It is never called per frame. Fallback: a rule-based planner.
- **Languages (confirmed):** Arabic (MSA), French, English; a few Darija phrases as a stretch goal.
- **Submission target (confirmed):** submit on **Oct 26, 18:00 Algeria time**. The official deadline (Oct 26, 11:59 pm PT ≈ Oct 27, 07:45 GMT+1) is buffer, not a goal.

**Decisions locked by this plan:**

- One hero flow: *register item → “find my X” → coached search → confirm → log last-seen*.
- Target 12 everyday classes, the ones most often lost by visually impaired people: keys, wallet, phone, glasses, sunglasses, remote, earbuds case, pill bottle or medicine box, cup or mug, bottle, bag or handbag, watch. Optional 13th and 14th: charger cable, white cane.
- Pursue **both** featured awards. COOL is a measurable side quest owned by B; the agent is core.

## Software Requirements Specification (SRS)

### 1. Purpose and scope

FindMine helps people with visual impairment find specific personal belongings indoors, using a phone camera, voice and audio/haptic guidance. A cloud agent runs the search and is built on OpenCV 5 and AWS. This SRS covers the competition release, v1.0, due Oct 26, 2026.

**Product promise:** *“Tell me what you lost. I’ll help you point the camera well, search systematically, and tell you honestly when I’ve found it — or where to look next.”*

### 2. Users and stakeholders

| User | Needs | Implication |
| --- | --- | --- |
| Blind user (primary) | Hands-free, speech-only, no visual UI dependence | Voice-first, screen-reader-labelled controls, audio + haptics |
| Low-vision user | Large high-contrast UI, magnified view | High-contrast theme, large buttons, optional zoomed preview |
| Sighted helper / family | Registers items, reviews history | Simple "add item" photo flow; history page |
| Judges | Try it in 2 minutes, verify claims | Public URL, demo mode with a pre-registered sample item, trace viewer, benchmark scripts |

### 3. Operating environment

- Client: Chrome on Android 10 or later (primary) and Safari on iOS 16 or later (reduced: no vibration or torch). HTTPS is required for camera access.
- Server: Docker container on an AWS Graviton (arm64) EC2 instance; Python 3.11+, OpenCV 5.x (COOL build where available), FastAPI + WebSocket.
- Cloud: S3, DynamoDB, Bedrock, Polly (optional), CloudWatch; deployed with AWS CDK or Terraform.

### 4. Functional requirements

Priority: **M** = must (MVP by Day 5), **S** = should (by Day 8 freeze), **C** = could (only if ahead).

**F1 — Item registration**

| ID | Requirement | Pri |
| --- | --- | --- |
| F1.1 | User or helper registers an item: name (spoken or typed), class from the 12-class list or "other", 3–5 photos from different angles | M |
| F1.2 | Capture guidance while registering: item centred, sharp, well lit; a bad photo is rejected with a spoken reason | M |
| F1.3 | System extracts and stores the item's fingerprint: DNN embedding, ORB/SIFT keypoints + descriptors, HSV colour histogram, and dominant colour name | M |
| F1.4 | User can list, rename, re-photograph and delete items; deleting removes photos and features from S3/DynamoDB | M |
| F1.5 | Free-text description used as a hint ("red keychain, Peugeot key") | C |

**F2 — Starting a search**

| ID | Requirement | Pri |
| --- | --- | --- |
| F2.1 | Voice command "find my \<item>" in AR, FR or EN via Web Speech API; matches saved item names with fuzzy matching | M |
| F2.2 | If the name is ambiguous or unknown, the system asks a clarifying question ("Your house keys or your car keys?") | S |
| F2.3 | Search by category for unregistered items ("find a cup") | M |
| F2.4 | Single large on-screen button + volume-key / double-tap shortcut to start, pause, resume and stop | M |

**F3 — Live perception (OpenCV 5)**

| ID | Requirement | Pri |
| --- | --- | --- |
| F3.1 | Detect target-class objects in frames with a fine-tuned detector run through `cv.dnn` | M |
| F3.2 | Verify instance identity: embedding cosine similarity + ORB/SIFT matching with RANSAC homography inliers + colour histogram, fused into one calibrated confidence | M |
| F3.3 | Track the confirmed object between detections (OpenCV tracker + Kalman smoothing) so guidance stays fluid at less than 200 ms | S |
| F3.4 | Tile/zoom search: when the frame is clean but nothing is found, re-run detection on high-resolution tiles to catch small objects (keys, earbuds) | S |
| F3.5 | Handle partial occlusion and objects cut by the frame edge: report "partly visible, move left" | S |
| F3.6 | Read a label to disambiguate (OCR on a medicine box, barcode/QR) with OpenCV text detection/recognition and barcode modules | C |

**F4 — Camera coach**

| ID | Requirement | Pri |
| --- | --- | --- |
| F4.1 | Per-frame quality metrics: blur (variance of Laplacian), exposure (histogram clipping, mean luminance), shake (optical-flow magnitude), lens obstruction (persistent low-texture dark region), out-of-focus | M |
| F4.2 | One spoken, actionable fix, rate-limited so it does not nag: "too dark — turning on the torch", "move slower", "wipe the lens" | M |
| F4.3 | Turn on the torch automatically (Android Chrome `torch` constraint) when the frame is dark and the user allowed it | S |
| F4.4 | Verify the fix: re-measure after the fix and say "better, continue" or escalate | S |

**F5 — Guidance**

| ID | Requirement | Pri |
| --- | --- | --- |
| F5.1 | Speak direction as clock positions ("at 2 o'clock") plus up/down | M |
| F5.2 | Continuous audio cue: stereo pan = horizontal offset, pitch/beep rate = closeness to the centre and size of the object (Web Audio API) | M |
| F5.3 | Haptic pulse that gets faster as the object is centred and near (Vibration API) | S |
| F5.4 | Coarse distance band (near / arm's length / far) from apparent size and known class size, reported as a band, never a precise number | S |
| F5.5 | Verbosity settings: short / detailed; speech rate; voice language | S |

**F6 — Agent (active search)**

| ID | Requirement | Pri |
| --- | --- | --- |
| F6.1 | Search state machine: COACH → SCAN → CANDIDATE → VERIFY → GUIDE → CONFIRM / NOT-FOUND, driven by vision events | M |
| F6.2 | Coverage memory: record which compass sectors (phone orientation sensor) and pitch bands were scanned with good-quality frames; direct the user to unscanned sectors | S |
| F6.3 | LLM planner (Bedrock) invoked on events (start, ambiguity, 20 s with no candidate, repeated quality failures) with tools: `detect`, `verify_instance`, `assess_quality`, `tile_search`, `set_torch`, `ask_user`, `speak`, `recall_last_seen`, `suggest_places` | S |
| F6.4 | Human-in-the-loop: below the high-confidence threshold, ask "I think I see your wallet at 1 o'clock — is it yours?" and never say "found" on its own | M |
| F6.5 | When several candidates match, describe and disambiguate ("two black wallets: the left one has a red tag") | S |
| F6.6 | Explain failure and suggest the next action ("I've scanned the desk and the shelf in good light; your keys were last seen in the kitchen at 14:02") | M |
| F6.7 | Every agent step is logged as a structured trace (inputs, OpenCV outputs, decision, action) and shown in a trace viewer | M |

**F7 — Memory**

| ID | Requirement | Pri |
| --- | --- | --- |
| F7.1 | When an item is confirmed, store the time, a room label (spoken by the user or picked from a list) and a thumbnail with faces blurred | M |
| F7.2 | "Where did I last see my keys?" answers from memory | S |
| F7.3 | Passive sightings: a registered item seen with high confidence during another search updates its last-seen | C |

**F8 — Safety, privacy, help**

| ID | Requirement | Pri |
| --- | --- | --- |
| F8.1 | Live frames are processed in memory and not stored by default; storage of evaluation frames is opt-in | M |
| F8.2 | Faces in any stored image are blurred (OpenCV `FaceDetectorYN` + Gaussian blur) | S |
| F8.3 | "Call my helper" button: shares the current frame or a link with a chosen contact via the OS share sheet | C |
| F8.4 | Spoken disclaimer on first run: not a mobility or navigation aid | M |

### 5. Non-functional requirements

| ID | Requirement | Target |
| --- | --- | --- |
| N1 | End-to-end guidance latency (frame captured → audio cue) | p50 ≤ 300 ms, p95 ≤ 600 ms on 4G/Wi-Fi to the EU region |
| N2 | Server throughput per `c7g.xlarge` | ≥ 10 fps of full pipeline (detect + verify) |
| N3 | Instance-verification precision at the "found" threshold | ≥ 95% (false "found" is the worst error) |
| N4 | Task success (item found within 2 min, volunteer test) | ≥ 80% in normal light, ≥ 60% in low light/clutter |
| N5 | Accessibility | Every control has an ARIA label; works with TalkBack; contrast ≥ 7:1; no gesture needed beyond a tap |
| N6 | Availability during judging (Oct 27 – Nov 9) | Instance kept running; health check + auto-restart; CloudWatch alarm |
| N7 | Security | HTTPS/WSS only; S3 private + SSE; per-device random ID (no accounts); least-privilege IAM; no secrets in repo |
| N8 | Reproducibility | `make deploy` / one CDK command; pinned `requirements.txt` with hashes; Docker image tag; seeds fixed for eval |
| N9 | Cost | Under $2/day while idle; documented cost per search-minute |
| N10 | Observability | Structured JSON logs, CloudWatch metrics (latency, fps, success, quality alerts), dashboard link in the report |

### 6. Out of scope for v1.0

Currency recognition, full document reading, outdoor navigation, obstacle/stair detection, face recognition of known people, native apps, user accounts, and live volunteer video calls. Mention them as "future work" in the report: they show you know the space without spending days on them.

## Architecture

&#91;embedded content: FindMine architecture · phone, vision service, AWS services\]

The phone is a thin sensor and speaker. All vision runs in one OpenCV 5 service on Graviton, so the COOL benchmark measures the real workload. The LLM is only called on events, which keeps per-frame latency deterministic.

### Per-frame pipeline (every frame, \~5 fps)

1. **Decode + downscale** (`cv.imdecode`, `cv.resize`).
2. **Camera coach** on a 320 px grey copy: blur = `cv.Laplacian(...).var()`; exposure = `cv.calcHist` share of pixels below 10 or above 245, plus mean; shake = median `cv.calcOpticalFlowPyrLK` displacement on `cv.goodFeaturesToTrack` points from the last frame; lens blockage = a large connected region (`cv.connectedComponentsWithStats` on a low-gradient mask) that stays the same for more than 1 s. Bad quality emits a `quality_event` and skips detection.
3. **Detect** with `cv.dnn.readNetFromONNX` (fine-tuned YOLO, 640 input) and `cv.dnn.NMSBoxes`. When the frame is clean and nothing is found, the agent can call **tile search**: 2×2 tiles at full resolution for small objects.
4. **Verify instance** for boxes of the target class: crop → embedding net via `cv.dnn` → cosine similarity to the item's reference embeddings; `cv.SIFT_create`/`cv.ORB_create` + `cv.BFMatcher` ratio test + `cv.findHomography(..., cv.RANSAC)` inlier count; `cv.compareHist` on HSV histograms. Fuse the three into one score with a logistic regression calibrated on the eval set, giving a confidence in \[0,1\].
5. **Track** the confirmed box with an OpenCV tracker (`cv.TrackerVit`/`TrackerNano` if the OpenCV 5 build ships them, else `TrackerCSRT` from contrib), with a `cv.KalmanFilter` on centre and size. Re-detect every N frames or when the tracker's score drops.
6. **Guidance math**: box centre → clock position and pan value; box height vs. the class's typical real size and the camera's horizontal FOV → distance band.

### Agent loop (events, not frames)

A deterministic **state machine** handles the real-time path: `COACH → SCAN → CANDIDATE → VERIFY → GUIDE → CONFIRM | NOT_FOUND`. The **LLM planner** (Bedrock, tool use) is invoked only at decision points. It gets a compact JSON observation (quality stats, candidates with confidences, the coverage map, last-seen, elapsed time) and returns the next tool call. Examples that count as Agentic Vision evidence:

- Dark frames → planner calls `set_torch(on)` → the coach re-measures → if still dark, `speak("Move closer to a lamp")`.
- Two candidates at 0.62 and 0.58 → planner calls `describe_candidates` (colour, relative position, any OCR text) → `ask_user("Left one with the red tag, or the right one?")`.
- 20 s, no candidate, coverage shows the right half unscanned → `speak("Turn slowly to your right")`; all sectors scanned in good light → `tile_search` on low pitch angles (floor), then `recall_last_seen` → `suggest_places`.

Every step writes a trace record `{t, observation, opencv_outputs, decision, tool, result}` to DynamoDB and CloudWatch. A small `/trace/<session>` page replays it. That page is your Agentic Award evidence and a strong video moment.

### AWS components

| Service | Use | Notes |
| --- | --- | --- |
| EC2 Graviton (`c7g.xlarge`) | Vision service + agent in Docker; Caddy for automatic HTTPS | Use an Elastic IP + a free domain or `sslip.io` hostname for TLS |
| COOL (AWS Marketplace) | OpenCV build optimised for Graviton | Subscribe on Day 1; fall back to stock OpenCV 5 arm64 |
| EC2 x86 (`c7i.xlarge`, short-lived) | Baseline for the COOL benchmark only | Terminate after measuring |
| Amazon Bedrock | Planner, clarifying questions, failure explanations | Enable model access on Day 1 (can take time) |
| S3 | Reference photos, model artefacts (ONNX), eval reports | Private, SSE-S3, lifecycle on eval frames |
| DynamoDB | Items, last-seen, search sessions, traces | On-demand capacity |
| CloudWatch | JSON logs, custom metrics, dashboard, alarm on health check | Screenshot it for the report |
| Polly (optional) | Neural voices for AR/FR if browser TTS sounds poor | Cache phrases in S3 |
| CDK or Terraform | One-command reproducible deploy | Required for the reproducibility score |

### Repository layout

```
findmine/
  app/          # PWA: index.html, app.js, audio.js, sw.js
  server/       # FastAPI: ws.py, pipeline/{coach,detect,verify,track,guide}.py, agent/
  ml/           # notebooks + scripts: data prep, training, export to ONNX, eval
  infra/        # CDK/Terraform stack, Dockerfile, Caddyfile
  bench/        # COOL vs baseline benchmark scripts + results
  eval/         # test protocol, results CSVs, failure gallery
  docs/         # report, architecture diagram, video script
```

## Data: sources, study and collection

Three different jobs need three different datasets. Don't try to make one dataset do everything.

| Job | Dataset | Why it fits | Watch out for |
| --- | --- | --- | --- |
| Detector training (12 classes) | **COCO 2017** (phone, cup, bottle, remote, handbag, backpack…) + **LVIS v1** (same images; adds key, wallet, eyeglasses, sunglasses, earphones, watch, pill bottle…) | Free, CC BY, already in YOLO-convertible format | LVIS labels are not exhaustive per image: only train an LVIS class on images where it is "verified" or you risk false negatives as background |
| Detector top-up | **Open Images V7** subset via FiftyOne for weak classes; Roboflow Universe sets for keys / earbuds / pill boxes | Fills rare classes quickly | Check each set's licence and image quality; drop near-duplicates |
| Instance-matching evaluation | **ORBIT** (Microsoft): \~3,800 videos of \~490 personal objects filmed by 77 blind and low-vision people, with "clean" and "clutter" videos | Exactly our task: few-shot recognition of *personal* items, filmed by the target users | Large: download a subset (e.g. 40 users). Check licence terms. Use it for evaluation and threshold calibration, not to train the embedding on test users |
| Camera-coach evaluation | **VizWiz image-quality-issues** dataset (photos taken by blind users, labelled blur, under/over-exposure, framing, obstruction…) | Real failure modes of our users, with labels | Labels are crowd votes: use majority vote. Use it to tune thresholds and report precision/recall |
| Real-world test set (ours) | **FindMine-Home**: phone videos you record in your own homes | Domain match (Algerian homes, your phones, real lighting) | Must be collected by Day 3. This is the set judges care most about |

### FindMine-Home collection protocol (both of you, Day 2–3, \~3 hours each)

- **Items:** each person picks 10 personal items from the 12 classes. Include **look-alike pairs** on purpose: two black wallets, two sets of keys, two similar mugs.
- **Reference capture:** 5 photos per item (front, back, two sides, in hand) on a plain surface, using the app's registration flow once it exists. Before then, use the phone camera at the same 640 px.
- **Search videos:** 20–30 s sweeps, 3 rooms × 3 conditions: *normal light*, *low light* (lamp only), *clutter* (item among other objects, partly hidden). Mix of distances (0.5 m, 1.5 m, 3 m). Also record the **look-alike distractor** next to the target.
- **Negative videos:** the same rooms *without* the target. These measure the false "found" rate.
- **Bad-camera clips:** deliberately shaky, finger on lens, dark, pointed at the ceiling. These are coach test cases.
- **Annotation:** sample 2 fps, label target boxes with CVAT or Roboflow (plus an "instance id" attribute). Aim for \~1,500 labelled frames. Faces: blur before anything leaves your laptop.
- **Consent:** only film your own homes; if other people appear, get permission or blur.
- **Split by room/video, never by frame.** Neighbouring frames are near-duplicates and would leak between train and test.

### Data study (Day 2–3, owner A) — a notebook that becomes a report figure

1. **Class balance:** instances per class after merging COCO + LVIS + top-ups. Target ≥ 1,500 instances per class; flag the weak ones (keys and earbuds will be).
2. **Object size:** the distribution of box area as a fraction of the image. Keys, earbuds and rings are tiny → justifies tile search and a larger input size for those classes.
3. **Domain gap:** compare brightness, blur (Laplacian variance) and box size between COCO/LVIS and FindMine-Home. Expect home video to be darker and blurrier → justifies augmentations and the coach.
4. **Quality thresholds:** on VizWiz, plot Laplacian variance and clipping ratio for "blur" vs "no blur" images (and exposure labels) and pick thresholds from ROC curves. This is a cheap, rigorous, very "OpenCV" result.
5. **ORBIT difficulty:** similarity distributions between same-object and different-object pairs, clean vs. clutter. Sets realistic expectations for the verifier.

Keep every chart from this notebook: they go straight into the technical report's evaluation section.

## Models, training and the improvement loop

**Golden rule:** train in PyTorch, but **run everything through OpenCV 5 at inference** (`cv.dnn` with ONNX). That is what "substantive OpenCV 5 use" means, and it makes the COOL benchmark meaningful.

### Model 1 — Detector (owner A)

- **Choice:** Ultralytics YOLO11-n (fast) and YOLO11-s (accurate); pick after measuring both on Graviton. Licence note: Ultralytics is AGPL-3.0. That is fine for a judge-accessible repo; state it in the report. Permissive fallback: YOLOX (Apache-2.0).
- **Day 1 spike (do this first!):** export the stock COCO model to ONNX (`opset=12`, static 640×640, `simplify=True`), run it with `cv.dnn` in OpenCV 5, and check the boxes match PyTorch on 50 images. If it doesn't load, try another opset or YOLOv8 before training anything.
- **Training run 1 (Day 3):** start from COCO weights; 12 classes; `imgsz=640`; 60–80 epochs; batch as large as the GPU fits; Colab/Kaggle T4 ≈ 2–4 h for the nano model on \~30k images. Augmentations that match our users: HSV jitter, **motion blur**, **low-light gamma (0.4–0.7) + noise**, random crops that cut objects at the frame edge, mosaic.
- **Training run 2 (Day 6–7):** fine-tune on FindMine-Home train split + hard negatives (frames from the negative videos where run 1 fired), low LR, 20–30 epochs. Optional: `imgsz=960` variant only for tile search on small classes.
- **Export + verify:** ONNX → `cv.dnn` parity check (mAP difference < 1 point) → upload to S3 with a version tag.

### Model 2 — Instance verifier (owner A)

- **Embedding:** a pretrained self-supervised ViT (e.g. DINOv2-small, Apache-2.0) exported to ONNX. Fallback if `cv.dnn` can't run it: a CNN backbone (MobileNetV3 / ResNet-50 global-pooled features). No training at first: these features are already strong for instance retrieval.
- **Geometry:** SIFT (`cv.SIFT_create`, in main since 4.4) for textured items; ORB for speed. Ratio test 0.75 → `cv.findHomography` RANSAC → inlier count and inlier ratio. Strong for keys, wallets with logos and medicine boxes; weak for plain mugs, which is where colour helps.
- **Colour:** HSV 2-D histogram (H×S, 30×32 bins) of the masked crop, `cv.compareHist(HISTCMP_CORREL)`, plus a named dominant colour for speech ("black", "red").
- **Fusion + calibration:** build same/different pairs from ORBIT (clean and clutter) and FindMine-Home. Fit a logistic regression on \[cosine, inliers, inlier ratio, hist corr\] and choose **two thresholds**: *found* (precision ≥ 95%) and *ask* (precision ≥ 70%). Below *ask*, stay silent and keep searching.
- **Stretch (only if Day 7 is green):** fine-tune the embedding with a metric-learning loss on ORBIT *train* users and report the gain on *test* users.

### Model 3 — Camera coach (no training)

Pure OpenCV metrics, with thresholds picked from VizWiz ROC curves and checked on our bad-camera clips. Being honest that "classic CV is enough here, and here is the ROC to prove it" is a strong technical-execution point for this panel.

### The improvement loop (Days 5–9)

Run this loop daily from Day 5. It is how the model gets better, and it produces the "evaluation evidence + failure cases" the rules require.

1. Run the **eval harness** (`make eval`) on FindMine-Home val + ORBIT subset + VizWiz subset. It writes one CSV and a markdown summary.
2. **Error analysis:** sample 30 failures and tag each one: *small object*, *dark*, *blur*, *occluded*, *look-alike FP*, *edge-cut*, *wrong class*.
3. **Fix the biggest bucket only:**
   - small object → tile search / higher input size
   - dark or blur → coach thresholds, augmentation, torch policy
   - look-alike FP → raise the *found* threshold, weight geometry more, ask-to-confirm
   - occluded or edge-cut → crop augmentation, "partly visible" message
   - wrong class → more data for that class, hard negatives
4. Re-run eval and log the delta in `eval/CHANGELOG.md`. This becomes the "what we improved and by how much" table in the report.

### Ablations to report (judges love these)

| Variant | What it shows |
| --- | --- |
| Category detector only (what competitors do) vs. + instance verifier | Value of "mine, not any" on look-alike pairs |
| Embedding only vs. + SIFT/RANSAC vs. + colour | Contribution of each OpenCV cue |
| Without vs. with tile search | Recall on small items |
| Coach off vs. on | Time-to-find and success in low light |
| Stock OpenCV x86 vs. OpenCV arm64 vs. COOL on Graviton | Latency, fps, $ per 1,000 frames |

## Evaluation plan

Three levels: component metrics (offline, automatic), system metrics (live, timed), and human outcomes (user tests). Report all three, **including the failures**.

### Level 1 — Component metrics (`make eval`, automatic)

| Component | Metric | Data | Target |
| --- | --- | --- | --- |
| Detector | mAP@50 and recall per class; recall for small objects | COCO/LVIS val subset + FindMine-Home test | mAP@50 ≥ 0.6 on Home test; report keys/earbuds separately |
| Instance verifier | Precision/recall at *found* and *ask* thresholds; top-1 accuracy among look-alikes | ORBIT clean + clutter subset, Home look-alike pairs | Precision ≥ 95% at *found* |
| Camera coach | Precision/recall per issue (blur, dark, bright, obstruction) | VizWiz quality subset + our bad-camera clips | F1 ≥ 0.75 for blur and dark |
| Distance band | Accuracy of near / arm / far | Home videos at 0.5, 1.5 and 3 m | ≥ 70% |

### Level 2 — System metrics (live, logged to CloudWatch)

- End-to-end latency p50/p95 (phone timestamp → cue played), server fps, CPU use, cost per search-minute.
- **COOL benchmark** (`bench/run.sh`): the same 1,000-frame replay through the full pipeline on (a) `c7i` x86 with stock OpenCV 5, (b) `c7g` Graviton with stock OpenCV 5 arm64, (c) `c7g` with COOL. Report per-stage latency (coach, detect, verify, track), throughput, CPU %, and $ per 1M frames, using on-demand prices. Repeat 5 runs and report mean ± std. Record the COOL version, AMI ID and instance type. Show proof that COOL is the library loaded: print `cv.getBuildInformation()` and save it.
- Agent metrics: share of sessions where the planner changed the action, tool-call success rate, Bedrock latency, and how often the user was asked to confirm.

### Level 3 — Human outcomes (Days 8–9)

- **Participants:** at least 6 blindfolded sighted volunteers (friends, classmates). **Blindfolded tests are the main study. As an optional bonus, get feedback from 1–2 visually impaired people**: contact a local association for the blind (in Algeria, for example, the national federation or a school for blind students) on Day 1, because they need time to say yes. Even one real user quote is gold for the impact score and the video.
- **Task:** "Find your \[item\]" in an unfamiliar room. 3 items × 2 conditions (normal, low light + clutter), with a look-alike distractor present in half the trials. 2-minute limit.
- **Conditions (within-subject, order counterbalanced):** (A) baseline = category-only detection with spoken direction, which is what competitor apps do; (B) full FindMine.
- **Measures:** success rate, time-to-find, wrong-item pickups, number of coach prompts, and a short **SUS** (System Usability Scale) questionnaire + 2 open questions.
- **Ethics:** informed consent form, no faces in published media without permission, the right to stop at any time; blindfolded participants always have a spotter.

### Failure-case gallery (required by the rules)

Keep a folder `eval/failures/` with 10–15 annotated examples: glass or transparent objects, very dark rooms, mirror reflections, identical items with no texture, objects inside bags, fast motion. For each one: what happened, why (based on the metrics), and what the system said to the user. **Honest failure reporting scores well; hidden failures discovered by a judge score terribly.**

## Day-by-day schedule (Oct 15 → Oct 26)

**A** = CV/ML track, **B** = App/Cloud/Agent track. Neither is a fixed person: claim tasks in the **FindMine Sprint Board** tracker app. Daily ritual: a 15-minute sync at the start (yesterday / today / blockers) and a **working demo on the deployed URL by the end of the day**. If a task overruns by more than half a day, cut scope; don't cut sleep.

Milestones: **M1** pipeline skeleton live (Oct 16) · **M2** end-to-end MVP "find my keys" (Oct 19) · **M3** feature freeze (Oct 22) · **M4** report + video done (Oct 25) · **M5** submitted (Oct 26, 18:00).

### Day 1 — Thu Oct 15 · Kickoff and de-risking

- [ ] Both: confirm the assumptions section, agree on names/roles, create the GitHub repo with the layout above, join the competition Slack, post a hello with #OpenCVComp26
- [ ] A: install OpenCV 5.x; **spike: YOLO ONNX in `cv.dnn`**; **spike: embedding ONNX (DINOv2-small) in `cv.dnn`**; record what works
- [ ] A: start the dataset downloads (COCO, LVIS annotations, ORBIT subset, VizWiz quality) — they take hours
- [ ] B: AWS account, billing alarm at $20/$50, region choice; **request Bedrock model access**; **subscribe to COOL on AWS Marketplace** and read its docs
- [ ] B: launch `c7g.xlarge`, Docker, Caddy with HTTPS on a hostname; "hello" FastAPI WebSocket
- [ ] Both: email/message 2–3 associations for blind people to recruit testers for Oct 22–23

### Day 2 — Fri Oct 16 · Skeleton end-to-end (M1)

- [ ] B: PWA camera → 640 px JPEG at 5 fps → WSS → server → JSON back; draw boxes + speak the class name; measure round-trip latency
- [ ] A: stock COCO YOLO via `cv.dnn` inside the server pipeline (phone, cup, bottle already work = instant demo)
- [ ] A: data study notebook part 1 (class balance, box sizes); write the merge script COCO+LVIS → YOLO format for the 12 classes
- [ ] Both (evening): record FindMine-Home session 1 in your own home (protocol above)

### Day 3 — Sat Oct 17 · Training run 1 + registration

- [ ] A: launch **detector training run 1** (Colab/Kaggle); meanwhile camera coach v1 (blur, exposure, shake) + VizWiz threshold ROC
- [ ] B: item registration flow (5 photos with capture guidance) → S3 + DynamoDB; feature extraction endpoint
- [ ] Both: finish FindMine-Home recording; start annotation (CVAT/Roboflow), split by room

### Day 4 — Sun Oct 18 · Verifier + guidance

- [ ] A: instance verifier v1 (embedding + SIFT/ORB RANSAC + HSV hist); build ORBIT and Home pairs; first precision/recall numbers
- [ ] B: guidance v1: clock-face speech, stereo-panned beeps (Web Audio), vibration; voice command "find my X" (Web Speech API, EN/FR/AR)
- [ ] A: export run-1 model → ONNX → `cv.dnn` parity check → S3

### Day 5 — Mon Oct 19 · MVP integration (M2)

- [ ] Both: wire it all: register → "find my keys" → coach → detect → verify → guide → confirm → save last-seen
- [ ] A: `make eval` harness v1 (writes CSV + summary); first full numbers
- [ ] B: state machine v1 with the ask-before-found rule; trace logging to DynamoDB/CloudWatch
- [ ] **Gate:** a teammate, blindfolded, finds their own keys among look-alikes. Record this on video (it's a backup demo clip). If the gate fails, Day 6 is only fixing the MVP

### Day 6 — Tue Oct 20 · Agent + tracking

- [ ] B: Bedrock planner with tool use (`set_torch`, `tile_search`, `ask_user`, `describe_candidates`, `recall_last_seen`, `suggest_places`); event triggers; trace viewer page
- [ ] B: coverage memory with DeviceOrientation (8 compass sectors × 3 pitch bands)
- [ ] A: tracker + Kalman between detections; tile search; distance bands
- [ ] A: hard-negative mining from negative videos → start **training run 2**

### Day 7 — Wed Oct 21 · Improve + COOL

- [ ] A: error analysis loop #1; calibrate fusion thresholds (logistic regression); export run 2; ablation runs
- [ ] B: **COOL benchmark** on x86 vs arm64 vs COOL (`bench/`); save build info as proof; then terminate the x86 box
- [ ] B: IaC (CDK/Terraform) for the whole stack; teardown + redeploy once from scratch to prove it
- [ ] Both: multilingual prompts (AR/FR/EN) + a few Darija phrases; verbosity setting

### Day 8 — Thu Oct 22 · Feature freeze (M3) + user tests

- [ ] Both: **freeze features at noon**. Only bug fixes after this
- [ ] Both: run the user-test protocol (blindfolded volunteers + visually impaired testers if recruited); film with consent
- [ ] A: error analysis loop #2 on the test logs
- [ ] B: accessibility pass with TalkBack; contrast check; big-button layout

### Day 9 — Fri Oct 23 · Fix + harden

- [ ] Both: fix the top 5 issues from user tests; finish any remaining tests
- [ ] B: CloudWatch dashboard + health-check alarm + auto-restart; privacy: face blur on stored thumbnails, delete-my-data endpoint
- [ ] A: final eval run on frozen models → all tables and charts; failure-case gallery

### Day 10 — Sat Oct 24 · Documentation

- [ ] A: technical report sections: problem, data, models, evaluation, limitations
- [ ] B: report sections: architecture, AWS deployment, COOL results, agent workflow + trace, responsible use; README with build/deploy/test steps; pinned dependencies
- [ ] Both: final architecture diagram + agent workflow diagram; storyboard the video
- [ ] Both: a friend who has never seen the project follows the README from scratch, and you fix whatever they get stuck on

### Day 11 — Sun Oct 25 · Video (M4)

- [ ] Both: film the demo scenes (morning light + evening low light); screen-record the trace viewer and dashboard
- [ ] Both: edit to ≤ 4:45; captions; upload as unlisted on YouTube
- [ ] Both: proofread the report; export a PDF; check every link from a logged-out browser

### Day 12 — Mon Oct 26 · Submit (M5)

- [ ] Morning: final full checklist (below); tag release `v1.0` in git
- [ ] Before 18:00 Algeria time: **submit** on the competition site
- [ ] Keep the server running through judging (Oct 27 – Nov 9); watch the alarm; don't push risky changes to the live URL
- [ ] Post the demo on social media with #OpenCVComp26 and tag OpenCV/AWS

## Risks and fallbacks

| Risk | Likelihood | Early signal | Fallback |
| --- | --- | --- | --- |
| A model won't load in `cv.dnn` (OpenCV 5 op support) | Medium | Day 1 spike fails | Other opset/simplify; YOLOv8 instead of YOLO11; CNN embedding instead of ViT |
| COOL setup is slow or unclear | Medium | Not running by Day 7 noon | Ship on stock OpenCV 5 arm64 and still report the x86 vs Graviton benchmark; drop the COOL award, not the project |
| Bedrock access delayed | Low–medium | No access by Day 3 | The state machine already works without the LLM; add the planner when access arrives. Rule-based fallback planner keeps the agent loop |
| Latency too high over mobile networks | Medium | p95 > 800 ms on Day 2 | Smaller frames (480 px), detection every 2nd frame + tracking; server in the closest EU region; on-device quality checks in JS for instant coach feedback |
| Keys and earbuds hard to detect | High | Low recall on Day 5 | Tile search, higher input size, more targeted data, and be honest about it in the limitations section |
| Look-alike false "found" | Medium | Precision < 95% | Raise the *found* threshold, always ask to confirm, weight geometry more |
| No visually impaired testers | Medium | No replies by Day 5 | Blindfolded volunteers + one remote interview with a blind user about the workflow; say so clearly in the report |
| Scope creep | High | Building anything outside the SRS before Day 7 | The owner of the other role says "no". Use the out-of-scope list |
| AWS bill surprise | Low | Billing alarm | Stop the x86 box after benchmarking; one Graviton instance; no NAT gateway; on-demand DynamoDB |
| Demo dies during judging | Low | Health alarm | Auto-restart, a pinned working Docker image, the recorded video, and an offer of a live screen-share in the submission |

**Zero-budget plan (paid only with free credits):** sign up for the AWS **Free plan**, which gives $100 in credits plus up to $100 more for completing activities. On the Free plan AWS does not bill you; when the credits or the 6-month period run out, the account stops instead of charging. Develop and train **locally and on Colab/Kaggle (free)**. Run the Graviton server only when needed, then keep a small one (`c7g.large` or `t4g.medium`) up for judging. Rough estimate: ≈ $5–10 of server time during dev, ≈ $15–30 for the server, public IP and disk through Nov 9, ≈ $1 for the x86 baseline, and a few dollars for Bedrock (a small model such as Haiku). Total ≈ **$30–50**, inside the $100 credits. Check the real prices in the AWS Pricing Calculator on Day 1, and set a $40 budget alert anyway.

## Submission kit

### Final checklist (every item is a rule requirement or a scoring lever)

- [ ] Technical report (PDF + markdown in repo): problem, users, architecture, OpenCV 5 implementation, AWS deployment, evaluation, limitations, responsible use
- [ ] Judge-accessible repo (public, or private with the judges invited) tagged `v1.0`
- [ ] Pinned dependencies (`requirements.txt` with exact versions, Docker image digest, OpenCV version printed at startup)
- [ ] Clear **build, deploy and test** instructions: `make setup`, `make deploy`, `make eval`, `make bench`
- [ ] Architecture diagram showing OpenCV 5, AWS, COOL and agent components
- [ ] **Agent workflow diagram** + an exported trace showing OpenCV output changing a later tool call (Agentic award)
- [ ] COOL evidence: version, instance type/AMI, benchmark method, inputs, baselines, results, `getBuildInformation()` proof (COOL award)
- [ ] Working public HTTPS URL + a **demo mode** (a pre-registered sample item and a printed test sheet, so judges can try it at their desk) + an offer of a live screen-share
- [ ] Unlisted video ≤ 5:00 showing the team, the app working, architecture and results
- [ ] Evaluation evidence with failure cases and limitations
- [ ] Award paths stated explicitly in the submission form: *Agentic Vision* and *Best Use of COOL*

### Technical report outline (8–12 pages)

1. **Problem and users:** a short story of a real person searching for keys; why category detection isn't enough; competitor gap table (from your research).
2. **Solution overview:** the three pillars and the hero flow.
3. **Architecture:** the diagram; per-frame pipeline; agent loop; AWS services.
4. **OpenCV 5 implementation:** each module and the exact APIs used; why classic CV (ORB/SIFT, RANSAC, Laplacian, optical flow, Kalman) sits next to DNN.
5. **Data:** sources, the FindMine-Home protocol, the data study charts.
6. **Training:** detector runs 1 and 2, verifier calibration.
7. **Evaluation:** component tables, ablations, COOL benchmark, user study (success, time-to-find, SUS, quotes).
8. **Agentic workflow:** state diagram, one annotated trace, failure handling, human control.
9. **Deployment, reproducibility and observability:** IaC, dashboard, cost.
10. **Responsible use:** privacy defaults, face blurring, ask-before-found, not a mobility aid, dataset licences, bias (lighting, item types, languages).
11. **Limitations and future work:** transparent objects, very dark rooms, on-device mode, currency/OCR, native apps.

### Video script (target 4:45)

| Time | Scene | Content |
| --- | --- | --- |
| 0:00–0:20 | Hook | A blindfolded hand sweeping a cluttered table; voice: "Every day, people who are blind lose time searching for things they own." |
| 0:20–0:40 | Team | Both of you on camera, names, roles, one line on why you're building it |
| 0:40–1:00 | The gap | "Apps can tell you *a* wallet is there. Not *your* wallet." Quick competitor contrast |
| 1:00–2:30 | Live demo | Register keys → "find my keys" in Arabic → coach: "too dark, torch on" → "turn right, you haven't scanned there" → two similar key sets → asks to confirm → hand guided in by beeps → "Found, saved: living room, 14:02" |
| 2:30–3:15 | How it works | Architecture diagram animation; trace viewer showing OpenCV output → agent decision → action |
| 3:15–4:00 | Results | 3–4 key numbers: precision at *found*, time-to-find A vs B, low-light success with/without coach, COOL speed-up and $ per 1M frames; one failure case |
| 4:00–4:30 | Impact | User test clip + quote; responsible-use bullets |
| 4:30–4:45 | Close | URL, repo, "Built with OpenCV 5 on AWS Graviton" |

### Habits that win hackathons

- **Demo first, then polish.** A boring flow that works every time beats an impressive one that works half the time.
- **Numbers in every claim.** "Better in low light" → "success went from 42% to 71% in low light with the coach (n = 24 trials)". Use *your* real numbers; the examples here are placeholders.
- **Make the judges' job easy:** a one-page "How to verify our claims" at the top of the README, with the exact commands and expected outputs.
- **Talk to the community:** share progress in the Slack and on social media with #OpenCVComp26. Organisers notice active teams.
