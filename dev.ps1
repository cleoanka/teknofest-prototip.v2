# AURA — geliştirme kısayolları (Windows / PowerShell 5.1+; Makefile eşleniği).
# Kullanım:
#   .\dev.ps1 setup                     # tek-komut kurulum (bootstrap.py --dev)
#   .\dev.ps1 run                       # tüm servisleri kaldır (run.ps1)
#   .\dev.ps1 test                      # unit testler (integration hariç)
#   .\dev.ps1 lint                      # ruff check
#   .\dev.ps1 format                    # black
#   .\dev.ps1 doctor                    # ortam/sağlık kontrolü (bağımlılık/cihaz/ağırlık/profil)
#   .\dev.ps1 train                     # eğitim CLI yardımı
#   .\dev.ps1 eval                      # örnek video + QoD A/B değerlendirmesi
#   .\dev.ps1 metrics                   # FTR §4 metrik raporu (eval_results\ab özetlerinden)
#   .\dev.ps1 video-test C:\yol\video.mp4   # gerçek video testi (annotated mp4 + JSON)
#   .\dev.ps1 clean                     # cache temizliği
# Her hedef Makefile'daki eşleniğiyle aynı komutu, aynı argümanlarla çalıştırır ve
# alt sürecin çıkış kodunu aynen döndürür (CI/parite için).
param(
  [Parameter(Position = 0)][string]$Target = "help",
  [Parameter(ValueFromRemainingArguments = $true)]$Rest
)
$ErrorActionPreference = "Stop"
# UTF-8: kutu çizimi/Türkçe karakterler cp1254/cp437 konsolunda bozulmasın.
$env:PYTHONUTF8 = "1"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
Push-Location $PSScriptRoot
$PY = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

# 'python' Microsoft Store stub'ı (WindowsApps) olabilir → 'py -3' launcher'ına düş.
# Store stub'ı sessizce Store'u açıp 9009 döndürür; bunu gerçek Python sanmayalım.
function Resolve-SystemPython {
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd -and $cmd.Source -notmatch 'WindowsApps') { return @{ Exe = "python"; Pre = @() } }
  if (Get-Command py -ErrorAction SilentlyContinue) { return @{ Exe = "py"; Pre = @("-3") } }
  return $null
}

try {
  # setup ve run kendi Python tespitlerini/bootstrap'larını yaptığından .venv ön
  # koşulundan muaf; diğer hedefler hazır bir .venv gerektirir.
  if ($Target -notin @("help", "setup", "run", "clean") -and -not (Test-Path $PY)) {
    Write-Error ".venv bulunamadı — önce .\setup.ps1 (veya .\dev.ps1 setup) çalıştırın"
    exit 1
  }
  switch ($Target) {
    "setup"      {
      # make setup eşleniği: bootstrap.py --dev (geliştirici ekstralarıyla).
      $sys = Resolve-SystemPython
      if (-not $sys) {
        Write-Error "Python 3.10+ bulunamadı ('python' veya 'py'). İndirme: https://www.python.org/downloads/"
        exit 1
      }
      # Splatting bir değişken gerektirir ($sys.Pre özelliği doğrudan @ ile açılmaz).
      $exe = $sys.Exe
      $pre = $sys.Pre
      & $exe @pre bootstrap.py --dev @Rest
      exit $LASTEXITCODE
    }
    "run"        {
      # make run eşleniği: tüm servisleri kaldıran run.ps1'i çağır (run.sh paritesi).
      # run.ps1 argüman almaz (run.sh gibi) — ek argüman aktarılmaz.
      & (Join-Path $PSScriptRoot "run.ps1")
      exit $LASTEXITCODE
    }
    "test"       { & $PY -m pytest -m "not integration" @Rest; exit $LASTEXITCODE }
    "lint"       { & $PY -m ruff check . @Rest; exit $LASTEXITCODE }
    "format"     { & $PY -m black . @Rest; exit $LASTEXITCODE }
    "doctor"     { & $PY tools/doctor.py @Rest; exit $LASTEXITCODE }
    "metrics"    { & $PY -m aura.eval --metrics-report --summaries eval_results/ab @Rest; exit $LASTEXITCODE }
    "train"      { & $PY -m train --help @Rest; exit $LASTEXITCODE }
    "eval"       {
      & $PY -m aura.eval --source data/samples/ornek.mp4 `
        --ground-truth data/samples/ornek_gt.json --qod-comparison @Rest
      exit $LASTEXITCODE
    }
    "video-test" {
      # make video-test VIDEO=... eşleniği: ilk argüman video yolu, kalanı ek
      # bayrak olarak test_video.py'ye aktarılır (ör. --max-frames 200 --no-video).
      $args2 = @($Rest)
      if ($args2.Count -lt 1 -or [string]::IsNullOrWhiteSpace($args2[0])) {
        Write-Error "kullanım: .\dev.ps1 video-test <video.mp4> [ek bayraklar]"
        exit 1
      }
      $video = $args2[0]
      $extra = if ($args2.Count -gt 1) { $args2[1..($args2.Count - 1)] } else { @() }
      & $PY tools/test_video.py --source $video --device auto @extra
      exit $LASTEXITCODE
    }
    "clean"      {
      # Makefile clean eşleniği: cache dizinleri + eval_results\*.json.
      Remove-Item -Recurse -Force .pytest_cache, .ruff_cache -ErrorAction SilentlyContinue
      Get-ChildItem -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force
      Get-ChildItem -Path eval_results -Filter "*.json" -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
      Write-Host "✓ cache temizlendi"
      exit 0
    }
    default      {
      Write-Host "Hedefler: setup | run | doctor | train | eval | metrics | test | lint | format | clean | video-test <video> | help"
      exit 0
    }
  }
} finally {
  Pop-Location
}
