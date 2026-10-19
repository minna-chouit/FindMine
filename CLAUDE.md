# FindMine: OpenCV AI Competition 2026 entry

FindMine is a phone web app (PWA) that helps blind and low-vision people find **their own** belongings, such as *my* keys rather than "a key". The phone streams camera frames to an OpenCV 5 vision service on AWS. That service does five things:

- **Coaches the camera:** detects blur, darkness, shake and a blocked lens.
- **Detects objects:** finds target-class objects with a YOLO model run through `cv.dnn`.
- **Checks it is the user's item:** compares the object to the item's 3–5 reference photos (embedding + SIFT/ORB with RANSAC + HSV colour histogram).
- **Guides the user:** clock-face speech, stereo beeps and vibration.
- **Runs the search as an agent:** a state machine, plus a Bedrock LLM planner that is called only on events.

Two-person team, no fixed leads. Read the whole plan before changing scope.

## Source of truth

- **Plan + SRS:** `planAndSRS.md` in this folder, exported from the Claude Doc https://claude.ai/code/artifact/9e3c6076-3de9-4594-87f5-e62d7ed8a741. Requirement IDs (F1.1, N3, …) come from it.
- **Sprint Board (tasks, shared live with the teammate):** https://claude.ai/artifact/7pGERszWonAca9u975Ky3a
  - Read it with the ArtifactData tool: `action: "query"`, `collection: "tasks"`. Each doc has `day` (5 = Oct 19 … 12 = Oct 26), `title`, `track` (A = CV/ML, B = App/Cloud/Agent, Both), `done`, `claimedBy`, `order`.
  - When the user picks a task, work on it. When it works **and has been tested**, tick it: `update` with `done: true`, `doneAt`, and the `if_version` you read. Don't tick work you haven't verified.
  - Ask before deleting or re-scheduling tasks.
  - The board page's source is `docs/sprint-board.html`. To change the page, edit that file and republish it with the Artifact tool, passing the board URL as `url` (otherwise you create a new board).

## Hard rules (competition requirements)

- **OpenCV 5 does the vision at inference time.** Every detection, matching, tracking and quality step goes through OpenCV 5 APIs: `cv.dnn` with ONNX models, `cv.SIFT_create`/`cv.ORB_create`, `cv.findHomography`, `cv.calcOpticalFlowPyrLK`, `cv.KalmanFilter`, and so on. PyTorch is only for training and ONNX export, never in the serving path.
- **A meaningful part runs on AWS:** the vision service on EC2 Graviton (arm64), S3, DynamoDB, Bedrock and CloudWatch. The AWS account is on the Free plan with credits only, and the budget is $0, so never create anything expensive (no NAT gateways, no GPU instances on AWS). Stop instances that aren't needed.
- **Pinned dependencies:** exact versions in `requirements.txt` (OpenCV 5.x pinned exactly). Log `cv2.__version__` at server start.
- **Reproducible:** clear build, deploy and test commands in the README. No secrets in the repo; use `.env`, which is git-ignored.
- **Training** runs on the team's NVIDIA platform GPU credit (or Colab/Kaggle), not on AWS.
- **Languages:** Arabic, French and English for speech and voice commands. A few Darija phrases are a stretch goal.
- **Accessibility first:** every control is labelled for screen readers, tap targets are large, contrast is high, and nothing depends only on visuals.
- **Safety:** never announce "found" below the *found* confidence threshold; ask the user to confirm instead. Live frames are not stored by default, and faces are blurred in anything that is stored.

## Repo layout (create folders as needed)

```
app/      PWA: index.html, app.js, audio.js
server/   FastAPI + WebSocket; pipeline/{coach,detect,verify,track,guide}.py; agent/
ml/       data prep, training, ONNX export, eval scripts
infra/    Dockerfile, Caddyfile, deploy script
bench/    COOL vs baseline benchmark
eval/     test protocol, results CSVs, failure gallery
docs/     report, diagrams, video script
```

## Working style

- Work on one sprint-board task at a time. Keep diffs small, test the task, then commit.
- Don't add libraries when OpenCV, the standard library or browser APIs already do the job.
- Non-trivial logic (quality thresholds, score fusion, clock-direction maths, the state machine) gets one small runnable test.
- Deadline: submit by **Oct 26, 18:00 Algeria time (GMT+1)**. Prefer a working, demoable flow over extra features.
