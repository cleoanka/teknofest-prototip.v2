# AURA — geliştirme kısayolları (Windows / PowerShell 5.1+; Makefile eşleniği).
# Kullanım:
#   .\dev.ps1 test                      # unit testler (integration hariç)
#   .\dev.ps1 lint                      # ruff check
#   .\dev.ps1 format                    # black
#   .\dev.ps1 doctor                    # ortam/sağlık kontrolü (bağımlılık/cihaz/ağırlık/profil)
#   .\dev.ps1 train                     # eğitim CLI yardımı
#   .\dev.ps1 eval                      # örnek video + QoD A/B değerlendirmesi
#   .\dev.ps1 metrics                   # FTR §4 metrik raporu (eval_results\ab özetlerinden)
#   .\dev.ps1 video-test C:\yol\video.mp4   # gerçek video testi (annotated mp4 + JSON)
#   .\dev.ps1 clean                     # cache temizliği
param(
  [Parameter(Position = 0)][string]$Target = "help",
  [Parameter(ValueFromRemainingArguments = $true)]$Rest
)
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
$PY = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
try {
  if ($Target -notin @("help", "clean") -and -not (Test-Path $PY)) {
    Write-Error ".venv bulunamadı — önce .\setup.ps1 --dev çalıştırın"
    exit 1
  }
  switch ($Target) {
    "test"       { & $PY -m pytest -m "not integration" @Rest; exit $LASTEXITCODE }
    "lint"       { & $PY -m ruff check . @Rest; exit $LASTEXITCODE }
    "format"     { & $PY -m black . @Rest; exit $LASTEXITCODE }
    "doctor"     { & $PY tools/doctor.py @Rest; exit $LASTEXITCODE }
    "metrics"    { & $PY -m aura.eval --metrics-report --summaries eval_results/ab @Rest; exit $LASTEXITCODE }
    "train"      { & $PY -m train --help; exit $LASTEXITCODE }
    "eval"       {
      & $PY -m aura.eval --source data/samples/ornek.mp4 `
        --ground-truth data/samples/ornek_gt.json --qod-comparison @Rest
      exit $LASTEXITCODE
    }
    "video-test" {
      if (-not $Rest) { Write-Error "kullanım: .\dev.ps1 video-test <video.mp4>"; exit 1 }
      & $PY tools/test_video.py --source @Rest --device auto
      exit $LASTEXITCODE
    }
    "clean"      {
      Remove-Item -Recurse -Force .pytest_cache, .ruff_cache -ErrorAction SilentlyContinue
      Get-ChildItem -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force
      Write-Host "✓ cache temizlendi"
      exit 0
    }
    default      {
      Write-Host "Hedefler: test | lint | format | doctor | train | eval | metrics | video-test <video> | clean"
      exit 0
    }
  }
} finally {
  Pop-Location
}
