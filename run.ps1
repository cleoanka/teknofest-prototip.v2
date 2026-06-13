# AURA — tüm servisleri kaldırır (Windows / PowerShell 5.1+).
# .venv yoksa önce bootstrap çağırır. Servis modülü yoksa uyarır ve atlar.
# Not: '0.0.0.0' bind'i ilk çalıştırmada Windows Defender Güvenlik Duvarı onayı
# isteyebilir (servis başına bir kez) — beklenen davranıştır, izin verin.
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

$PY = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PY)) {
  Write-Host "▶ .venv bulunamadı — bootstrap çalıştırılıyor"
  # 'python' Store stub'ı olabilir → 'py -3' launcher'ına düş; sonucu DOĞRULA
  # (eskiden başarısız bootstrap sessizce yutulup 41. satırda kafa karıştıran
  # '.venv\Scripts\python.exe bulunamadı' hatasıyla patlıyordu).
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd -and $cmd.Source -notmatch 'WindowsApps') { python bootstrap.py }
  elseif (Get-Command py -ErrorAction SilentlyContinue) { py -3 bootstrap.py }
  else { Write-Error "Python bulunamadı ('python'/'py'). https://www.python.org/downloads/"; exit 1 }
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $PY)) {
    Write-Error "bootstrap başarısız — .venv oluşturulamadı (yukarıdaki logları inceleyin)"
    exit 1
  }
}

# bootstrap'ın ürettiği .env varsa AURA_* değişkenlerini yükle (kullanıcının
# oturumda elle set ettiği değerler ezilmez — önce gelen kazanır).
$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $envFile) {
  Get-Content $envFile | ForEach-Object {
    if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)=(.*)$' -and -not (Test-Path "env:$($Matches[1])")) {
      Set-Item -Path "env:$($Matches[1])" -Value $Matches[2].Trim()
    }
  }
}

$inferPort = if ($env:AURA_INFERENCE_PORT) { $env:AURA_INFERENCE_PORT } else { 8080 }
$qodPort   = if ($env:AURA_QOD_MOCK_PORT)  { $env:AURA_QOD_MOCK_PORT }  else { 8081 }
$nvPort    = if ($env:AURA_NV_MOCK_PORT)   { $env:AURA_NV_MOCK_PORT }   else { 8082 }

$procs = @()

function Clear-Port($port) {
  # Önceki çalıştırmadan kalıp bu portu hâlâ tutan dinleyiciyi serbest bırak;
  # aksi halde uvicorn "[10048] yalnızca bir kullanıma izin veriliyor" ile çöker.
  $owners = @()
  try {
    $owners = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction Stop |
                ForEach-Object { $_.OwningProcess } | Sort-Object -Unique)
  } catch {
    $owners = @(netstat -ano | Select-String ":$port\s" | Where-Object { $_ -match 'LISTENING' } |
                ForEach-Object { ($_.ToString().Trim() -split '\s+')[-1] } | Sort-Object -Unique)
  }
  foreach ($procId in $owners) {
    if ($procId -and $procId -ne 0) {
      try {
        Stop-Process -Id ([int]$procId) -Force -ErrorAction Stop
        Write-Host "  ⚠ Port $port önceki süreçten (pid $procId) serbest bırakıldı"
      } catch {}
    }
  }
}

function Test-PyModule($mod) {
  # PS 5.1 tuzağı: $ErrorActionPreference='Stop' + native stderr yönlendirmesi her
  # stderr satırını terminating NativeCommandError'a çevirir — modül-yok durumunda
  # python traceback basar ve 'uyar-ve-atla' dalı asla çalışmazdı. Probu geçici
  # 'Continue' ile sarıyoruz; karar yalnızca çıkış koduna bakar.
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  & $PY -c "import importlib; importlib.import_module('$mod')" 2>$null
  $found = ($LASTEXITCODE -eq 0)
  $ErrorActionPreference = $prev
  return $found
}

function Start-Svc($name, $app, $port) {
  $mod = $app.Split(":")[0]
  if (Test-PyModule $mod) {
    Clear-Port $port
    $p = Start-Process -FilePath $PY -ArgumentList @("-m","uvicorn",$app,"--host","0.0.0.0","--port",$port) -PassThru -NoNewWindow
    $script:procs += $p
    Write-Host "  ✓ $name → http://localhost:$port  (pid $($p.Id))"
  } else {
    Write-Host "  ⚠ $name modülü henüz yok ($app) — sonraki milestone'da gelir"
  }
}

try {
  Write-Host "▶ AURA servisleri başlatılıyor"
  Start-Svc "QoD mock"      "services.qod_mock.main:app"      $qodPort
  Start-Svc "NV mock"       "services.nv_mock.main:app"       $nvPort
  Start-Svc "Inference API" "services.inference_api.main:app" $inferPort
  Write-Host ""
  Write-Host "  Dashboard:  http://localhost:$inferPort/"
  Write-Host "  OpenAPI:    http://localhost:$inferPort/docs"
  Write-Host "  (Ctrl-C ile durdurun)"
  if ($procs.Count -gt 0) {
    # -EA SilentlyContinue: bir servis erken çökmüşse Wait-Process 'süreç yok'
    # hatasıyla finally'e atlayıp SAĞLIKLI servisleri de öldürmesin.
    Wait-Process -Id ($procs | ForEach-Object { $_.Id }) -ErrorAction SilentlyContinue
  } else {
    Write-Host "  ⚠ hiçbir servis başlatılamadı — bootstrap/uvicorn loglarını kontrol edin"
    exit 1
  }
} finally {
  Write-Host "`n▶ servisler durduruluyor..."
  foreach ($p in $procs) { try { Stop-Process -Id $p.Id -Force } catch {} }
  Pop-Location
}
