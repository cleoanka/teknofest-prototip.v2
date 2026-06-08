# AURA -- tum servisleri kaldirir (Windows / PowerShell 5.1+).
# .venv yoksa once bootstrap cagririr. Servis modulu yoksa uyarir ve atlar.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$PY = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PY)) {
  Write-Host "> .venv bulunamadi -- bootstrap calistiriliyor"
  python bootstrap.py
}

$inferPort = if ($env:AURA_INFERENCE_PORT) { $env:AURA_INFERENCE_PORT } else { 8080 }
$qodPort   = if ($env:AURA_QOD_MOCK_PORT)  { $env:AURA_QOD_MOCK_PORT }  else { 8081 }
$nvPort    = if ($env:AURA_NV_MOCK_PORT)   { $env:AURA_NV_MOCK_PORT }   else { 8082 }

$procs = @()
function Start-Svc($name, $app, $port) {
  $mod = $app.Split(":")[0]
  & $PY -c "import importlib; importlib.import_module('$mod')" 2>$null
  if ($LASTEXITCODE -eq 0) {
    $p = Start-Process -FilePath $PY -ArgumentList @("-m","uvicorn",$app,"--host","0.0.0.0","--port",$port) -PassThru -NoNewWindow
    $script:procs += $p
    Write-Host "  [OK] $name -> http://localhost:$port  (pid $($p.Id))"
  } else {
    Write-Host "  [!!] $name modulu henuz yok ($app) -- sonraki milestone'da gelir"
  }
}

try {
  Write-Host "> AURA servisleri baslatiliyor"
  Start-Svc "QoD mock"      "services.qod_mock.main:app"      $qodPort
  Start-Svc "NV mock"       "services.nv_mock.main:app"       $nvPort
  Start-Svc "Inference API" "services.inference_api.main:app" $inferPort
  Write-Host ""
  Write-Host "  Dashboard:  http://localhost:$inferPort/"
  Write-Host "  OpenAPI:    http://localhost:$inferPort/docs"
  Write-Host "  (Ctrl-C ile durdurun)"
  Wait-Process -Id ($procs | ForEach-Object { $_.Id })
} finally {
  Write-Host "`n> servisler durduruluyor..."
  foreach ($p in $procs) { try { Stop-Process -Id $p.Id -Force } catch {} }
}
