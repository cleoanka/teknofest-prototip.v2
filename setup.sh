#!/usr/bin/env bash
# AURA kurulum sarmalayıcı (macOS/Linux). Tüm mantık bootstrap.py'de.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 bootstrap.py "$@"
