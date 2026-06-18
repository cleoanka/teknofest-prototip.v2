#!/usr/bin/env bash
# AURA — tüm servisleri kaldırır (inference :8080, qod :8081, nv :8082).
# .venv yoksa önce bootstrap çağırır. Servis modülü henüz yoksa (erken milestone)
# uyarır ve atlar; mevcut olanları başlatır.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "▶ .venv bulunamadı — bootstrap çalıştırılıyor"
  python3 bootstrap.py
  if [ ! -x "$PY" ]; then
    echo "✗ bootstrap başarısız — .venv oluşturulamadı" >&2
    exit 1
  fi
fi

# bootstrap'ın ürettiği .env varsa AURA_* değişkenlerini yükle (oturumda elle
# set edilen değerler ezilmez — run.ps1 ile aynı sözleşme).
if [ -f "$ROOT/.env" ]; then
  while IFS='=' read -r k v; do
    case "$k" in ''|\#*) continue ;; esac
    v="${v%$'\r'}"
    [ -z "${!k:-}" ] && export "$k=$v"
  done < "$ROOT/.env"
fi

INFER_PORT="${AURA_INFERENCE_PORT:-8080}"
QOD_PORT="${AURA_QOD_MOCK_PORT:-8081}"
NV_PORT="${AURA_NV_MOCK_PORT:-8082}"

pids=()
start() { # ad  module:app  port
  local name="$1" app="$2" port="$3" mod="${2%%:*}"
  if "$PY" -c "import importlib; importlib.import_module('$mod')" >/dev/null 2>&1; then
    "$PY" -m uvicorn "$app" --host 0.0.0.0 --port "$port" &
    pids+=($!)
    echo "  ✓ $name → http://localhost:$port  (pid $!)"
  else
    echo "  ⚠ $name modülü henüz yok ($app) — sonraki milestone'da gelir"
  fi
}

cleanup() {
  echo; echo "▶ servisler durduruluyor..."
  for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done
}
trap cleanup INT TERM EXIT

# Sunucu dağıtımı: AURA_PROFILE=server ./run.sh  → config/profiles/server.yaml uygulanır
# (inference servisi load_config() içinde AURA_PROFILE env'ini otomatik okur).
echo "▶ AURA servisleri başlatılıyor  (profil: ${AURA_PROFILE:-varsayılan})"
start "QoD mock"      "services.qod_mock.main:app"      "$QOD_PORT"
start "NV mock"       "services.nv_mock.main:app"       "$NV_PORT"
start "Inference API" "services.inference_api.main:app" "$INFER_PORT"

echo
echo "  Dashboard:  http://localhost:$INFER_PORT/"
echo "  OpenAPI:    http://localhost:$INFER_PORT/docs"
echo "  (Ctrl-C ile durdurun)"
wait
