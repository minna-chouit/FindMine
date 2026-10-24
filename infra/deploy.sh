#!/usr/bin/env bash
# One-command deploy of FindMine on an EC2 instance (Ubuntu 22.04/24.04 or Amazon Linux 2023, x86 or Graviton).
#
#   git clone https://github.com/minna-chouit/FindMine.git && cd FindMine && bash infra/deploy.sh
#   (re-run it after every `git pull`: it is safe to run again)
#
# Needs: the instance's IAM role allows S3 (bucket), DynamoDB (table) and bedrock:InvokeModel;
# security group open on 80 and 443; the ONNX models uploaded once to s3://$FINDMINE_BUCKET/models/.
# Optional settings in .env (created on first run): FINDMINE_BUCKET, FINDMINE_TABLE, AWS_REGION, FINDMINE_DOMAIN.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
CADDY_VERSION=2.10.2

echo "== 1/6 system packages"
if command -v apt-get >/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq git curl python3-venv python3-pip libgl1 libglib2.0-0 >/dev/null
  PY=python3
elif command -v dnf >/dev/null; then
  sudo dnf install -y -q git python3.12 python3.12-pip mesa-libGL glib2 >/dev/null || sudo dnf install -y -q git python3.11 python3.11-pip mesa-libGL glib2 >/dev/null
  PY=$(command -v python3.12 || command -v python3.11)
else
  echo "unsupported OS: use Ubuntu or Amazon Linux 2023" >&2; exit 1
fi
"$PY" -c 'import sys; assert sys.version_info >= (3, 11), "need Python 3.11+"'

echo "== 2/6 settings (.env)"
TOKEN=$(curl -s -m 2 -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60" || true)
IP=$(curl -s -m 2 -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4 || true)
if [ ! -f .env ]; then
  cat > .env <<ENV
AWS_REGION=us-east-1
FINDMINE_BUCKET=findmine-items-bucket
FINDMINE_TABLE=findmine-items
FINDMINE_DOMAIN=${IP//./-}.sslip.io
ENV
fi
set -a; . ./.env; set +a
: "${FINDMINE_DOMAIN:?set FINDMINE_DOMAIN in .env (no public IP found)}"

echo "== 3/6 Python environment"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt

echo "== 4/6 models from s3://$FINDMINE_BUCKET/models/"
.venv/bin/python - <<PY
import os, boto3
s3 = boto3.client("s3", region_name=os.environ["AWS_REGION"])
for name in ("findmine12.onnx", "dinov2_vits14.onnx"):
    path = f"models/{name}"
    if not os.path.exists(path):
        s3.download_file(os.environ["FINDMINE_BUCKET"], path, path)
        print("  downloaded", path)
PY
# Face model for blurring stored photos: public, straight from OpenCV's model zoo (no S3 upload needed).
# Downloaded to .part first: a cut-off download must never leave a broken model that crashes every restart.
if [ ! -f models/face_detection_yunet_2023mar.onnx ]; then
  curl -sfL -o models/face_detection_yunet_2023mar.onnx.part \
    https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
  mv models/face_detection_yunet_2023mar.onnx.part models/face_detection_yunet_2023mar.onnx
  echo "  downloaded models/face_detection_yunet_2023mar.onnx"
fi

echo "== 5/6 service (systemd: starts on boot, restarts on crash, logs in journald)"
sudo tee /etc/systemd/system/findmine.service >/dev/null <<UNIT
[Unit]
Description=FindMine vision service
After=network-online.target

[Service]
User=$(id -un)
WorkingDirectory=$REPO
EnvironmentFile=$REPO/.env
ExecStart=$REPO/.venv/bin/python -m uvicorn --app-dir server main:app --host 127.0.0.1 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable -q findmine
sudo systemctl restart findmine

echo "== 6/6 HTTPS (Caddy) for https://$FINDMINE_DOMAIN"
if ! command -v caddy >/dev/null; then
  ARCH=$(uname -m); ARCH=${ARCH/x86_64/amd64}; ARCH=${ARCH/aarch64/arm64}
  curl -sL "https://github.com/caddyserver/caddy/releases/download/v$CADDY_VERSION/caddy_${CADDY_VERSION}_linux_$ARCH.tar.gz" | sudo tar -xz -C /usr/local/bin caddy
  sudo tee /etc/systemd/system/caddy.service >/dev/null <<UNIT
[Unit]
Description=Caddy
After=network-online.target

[Service]
ExecStart=/usr/local/bin/caddy run --config /etc/caddy/Caddyfile
ExecReload=/usr/local/bin/caddy reload --config /etc/caddy/Caddyfile
AmbientCapabilities=CAP_NET_BIND_SERVICE
Restart=always

[Install]
WantedBy=multi-user.target
UNIT
  sudo systemctl daemon-reload
  sudo systemctl enable -q caddy
fi
sudo mkdir -p /etc/caddy
[ -f /etc/caddy/Caddyfile ] && sudo cp -n /etc/caddy/Caddyfile /etc/caddy/Caddyfile.before-findmine
APP_DIR="$REPO/app"; [ -d "$APP_DIR" ] || APP_DIR="$REPO/infra/no-app"   # never serve the repo root (.env!)
mkdir -p "$REPO/infra/no-app"
sed "s#DOMAIN#$FINDMINE_DOMAIN#; s#APP_DIR#$APP_DIR#" infra/Caddyfile | sudo tee /etc/caddy/Caddyfile >/dev/null
sudo systemctl restart caddy

echo "== checking https://$FINDMINE_DOMAIN/health"
for i in $(seq 1 30); do
  if out=$(curl -sf -m 5 "https://$FINDMINE_DOMAIN/health"); then
    echo "  $out"; echo "DEPLOYED: https://$FINDMINE_DOMAIN  (app: wss://$FINDMINE_DOMAIN/ws, traces: /trace)"; exit 0
  fi
  sleep 3
done
echo "health check failed: see  sudo journalctl -u findmine -n 50  and  sudo journalctl -u caddy -n 50" >&2
exit 1
