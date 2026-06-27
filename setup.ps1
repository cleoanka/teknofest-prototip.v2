# RoadGuard kurulum sarmalayıcı (Windows / PowerShell 5.1+). Tüm mantık bootstrap.py'de.
# Geliştirici ekstraları (pytest/ruff/black) için: .\setup.ps1 --dev   (make setup eşdeğeri)
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
# Bazı hostlarda (redirected/ISE) OutputEncoding ataması istisna atar — yutalım.
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
Push-Location $PSScriptRoot
try {
  # 'python' PATH'te yoksa ya da Microsoft Store stub'ıysa 'py -3' launcher'ına düş
  # (Store stub'ı sessizce Store'u açıp 9009 döndürür — kurulum başarılı sanılırdı).
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd -and $cmd.Source -notmatch 'WindowsApps') {
    $exe = "python"; $preArgs = @()
  } elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $exe = "py"; $preArgs = @("-3")
  } else {
    Write-Error "Python 3.10+ bulunamadı ('python' veya 'py' launcher). İndirme: https://www.python.org/downloads/"
    exit 1
  }
  & $exe @preArgs bootstrap.py @args
  # Çıkış kodu YUTULMAZ: bootstrap başarısızsa (ör. smoke test) bunu aynen raporla.
  exit $LASTEXITCODE
} finally {
  Pop-Location
}
