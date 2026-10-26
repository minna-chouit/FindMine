#!/usr/bin/env bash
# Auto-update: every minute, pull new commits on main. Restart only if server code changed.
# If the new version does not answer /health within 60 s, roll back to the previous commit.
set -u
cd /home/ubuntu/FindMine || exit 1
git fetch -q origin main || exit 0
OLD=$(git rev-parse HEAD); NEW=$(git rev-parse origin/main)
[ "$OLD" = "$NEW" ] && exit 0
if ! git diff --quiet || ! git diff --cached --quiet; then echo "local changes on the server, not updating"; exit 0; fi
CHANGED=$(git diff --name-only "$OLD" "$NEW")
git merge -q --ff-only origin/main || { echo "main is not a fast-forward, not updating"; exit 0; }
echo "updated $OLD -> $NEW"
if echo "$CHANGED" | grep -qx 'requirements.txt'; then .venv/bin/python -m pip install -q -r requirements.txt; fi
if ! echo "$CHANGED" | grep -Ev '^(app/|docs/|eval/)|\.md$' | grep -q .; then
  echo "only app/docs changed: live without restart"; exit 0
fi
sudo systemctl restart findmine
for i in $(seq 1 12); do sleep 5; curl -fs localhost:8000/health >/dev/null && { echo "restarted, healthy"; exit 0; }; done
echo "ROLLBACK: new version not healthy, going back to $OLD"
git reset -q --hard "$OLD"
sudo systemctl restart findmine
