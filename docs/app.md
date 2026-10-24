# FindMine app (PWA)

FindMine is a phone web app that helps blind and low-vision people find **their own** things. It runs in the phone's browser and can be installed like an app. All the vision work happens on the server; the phone only sends camera pictures and does what the server says.

## One link

| Where | Link | Notes |
|---|---|---|
| **Main (server)** | `https://54-90-98-230.sslip.io/` | Pages and API on the same address. Use this one for demos and judges. |
| Backup (GitHub Pages) | `https://minna-chouit.github.io/FindMine/` | Redirects to `app/`, talks to the server above. |

Everything starts from the home page. No other link is needed.

## Pages

| Page | File | What it does |
|---|---|---|
| Home | `app/index.html` | Tap anywhere → speaks the safety notice (first time) and a welcome → listens: "find my X" or "register". Big buttons for everything. |
| Find my item | `app/search.html` | Pick an item (or arrive from voice) → camera → follows the server: speech, Yes/No, stereo beeps, vibration, torch, **Got it** saves where it was found. |
| Ask by voice | `app/find.html` | "find my keys" in EN / FR / AR (+ some Darija words) → opens the search for that item. |
| Register an item | `app/register.html` | Name (typed or 🎤 spoken) + type + optional training consent → 5 spoken sides (front, back, left, right, top). Photos are taken **automatically** when the picture is good (server camera coach + an object in view), with rising beeps; the button still works. |
| Settings | `app/settings.html` | Language, short/detailed speech, **text size**, **light/dark**, test mode (baseline = condition A), replay the safety notice, server address (team). |
| Camera test | `app/tools/camera.html` | Team tool: every box the server sees + round-trip speed. |
| Guidance test | `app/tools/guide.html` | Team tool: guide to any object type, no registration. |

## How a search works

```
Home ──"find my bottle"──▶ Ask by voice ──▶ Find my item
                                              │ camera frames (5/s, 640 px JPEG) over wss://…/ws
                                              ▼
                     server: coach → detect (YOLO, cv.dnn) → verify (DINOv2 + SIFT + colour) → state machine
                                              │ actions: speak · ask_user · guide · torch · save_last_seen
                                              ▼
                     phone: speaks, shows Yes/No, beeps left/right, vibrates, "Got it" → last-seen saved
```

Box choice: the detector proposes, the verifier decides. Boxes of the item's class are verified first; if there are none, the top boxes of **any** class are verified too (the detector often names small items wrong, e.g. keys → "phone"). `FINDMINE_ANY_CLASS=0` turns this off; the baseline test mode never uses it.

States: `SCAN → CANDIDATE / CONFIRM → GUIDE → FOUND` (or `COACH` for a bad picture, `NOT_FOUND`). "Found" is only said at match ≥ 0.8 or after the user says yes.

## Files

```
app/
  index.html  search.html  find.html  register.html  settings.html
  tools/camera.html  tools/guide.html
  css/app.css          design system (colours, sizes, buttons) – change the look here only
  js/config.js         server address + text size + light/dark, loaded first on every page
  js/prompts.js        every spoken sentence (EN, FR, AR, Darija) + safety notice + test mode
  assets/logo.svg      wordmark, indigo ink + yellow sparkle (light theme)
  assets/logo-dark.svg wordmark, pale ink + yellow sparkle (dark theme)
  icons/icon.svg       app icon source (f + sparkle on indigo)
  icons/icon-192.png   app icons, made from icon.svg
  icons/icon-512.png
  manifest.webmanifest sw.js   make it installable
index.html             (repo root) redirects to app/ for GitHub Pages
```

The server address is decided in `js/config.js`: when the server hosts the app, pages use their own address; anywhere else they use `DEFAULT_SERVER`. After a redeploy to a new IP, change `DEFAULT_SERVER` (or set it per phone in Settings → Team).

## Serving the app from the server

`infra/deploy.sh` already serves `app/` with Caddy, next to the API. On the current server (Caddy from apt, server in tmux), after `git pull`:

```
chmod o+x /home/ubuntu            # lets Caddy read the app folder
sudo tee /etc/caddy/Caddyfile > /dev/null <<'EOF'
54-90-98-230.sslip.io {
	encode gzip
	@api path /ws /health /items /items/* /trace /trace/*
	handle @api {
		reverse_proxy 127.0.0.1:8000
	}
	handle {
		root * /home/ubuntu/FindMine/app
		file_server
	}
}
EOF
sudo systemctl reload caddy
```

## Logo and icons

| File | Size | Notes |
|---|---|---|
| `app/assets/logo.svg` | vector | Hand-drawn bubbly wordmark. Indigo `#1E2252`; the sparkle on the second i is always butter `#FFDE7A`. |
| `app/assets/logo-dark.svg` | vector | Same, with pale ink `#F3F1FF` for dark mode. Pages show it with the `.logo-light` / `.logo-dark` classes. |
| `app/icons/icon-512.png` | 512×512 | The f + sparkle on indigo, inside the middle 80 % (Android crops the edges). |
| `app/icons/icon-192.png` | 192×192 | Same design, smaller. |

## Design tokens

| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` | `#FFF8EA` cream | `#14162E` | page, tap-to-start screen |
| `--text` | `#1E2252` indigo | `#F3F1FF` | all text |
| `--primary` | `#FFDE7A` butter | `#FFD86B` | main buttons |
| `--secondary` | `#DCE1FF` periwinkle | `#B7C0FF` | second buttons |
| `--sparkle` | `#FFDE7A` butter | `#FFDE7A` | the sparkle in the logo |
| `--danger` / `--success` | `#9B1C12` / `#1E6630` | `#FFAE9F` / `#9EE0AF` | Stop / No, Yes / Got it |

Font: **Atkinson Hyperlegible** (Braille Institute, made for low-vision readers) for all UI text; the bubbly lettering is only in the logo. See `docs/accessibility.md` for the checks.
