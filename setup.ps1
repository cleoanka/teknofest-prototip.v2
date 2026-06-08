# AURA kurulum sarmalayıcı (Windows / PowerShell 7+). Tüm mantık bootstrap.py'de.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
python bootstrap.py @args
