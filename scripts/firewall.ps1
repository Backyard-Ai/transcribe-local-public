<#
Optional second lock: tells Windows Firewall to stop this program's Python from
reaching the internet. The program already refuses connections from the inside.

Run from an administrator PowerShell window:
  scripts\firewall.ps1 on       adds the block
  scripts\firewall.ps1 off      removes it, for installing updates or new models
  scripts\firewall.ps1 status   shows the current rules

Windows starts a virtual environment's Python through the main Python install, so
the block also covers that install. Other programs using it are blocked as well.
#>
param([Parameter(Mandatory)][ValidateSet('on', 'off', 'status')][string]$Action)

# Without this, a failed rule would still print the success message below.
$ErrorActionPreference = 'Stop'

$identity = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if ($Action -ne 'status' -and -not $identity.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Output 'Nothing changed. Open PowerShell with "Run as administrator" and try again.'
    exit 1
}

$prefix = 'Transcribe Local - block '
$root = Split-Path $PSScriptRoot -Parent
$config = Get-Content (Join-Path $root '.venv\pyvenv.cfg') | Where-Object { $_ -match '^home\s*=' }
$base = ($config -split '=', 2)[1].Trim()
$programs = @(
    (Join-Path $root '.venv\Scripts\python.exe'),
    (Join-Path $root '.venv\Scripts\pythonw.exe'),
    (Join-Path $base 'python.exe'),
    (Join-Path $base 'pythonw.exe')
)

if ($Action -eq 'status') {
    Get-NetFirewallRule -DisplayName "$prefix*" -ErrorAction SilentlyContinue |
        Select-Object DisplayName, Enabled, Direction, Action | Format-Table -AutoSize
    return
}

Get-NetFirewallRule -DisplayName "$prefix*" -ErrorAction SilentlyContinue | Remove-NetFirewallRule
if ($Action -eq 'on') {
    foreach ($program in $programs) {
        New-NetFirewallRule -DisplayName "$prefix$program" -Direction Outbound -Program $program -Action Block | Out-Null
    }
    Write-Output "Blocked $($programs.Count) programs from reaching the internet."
} else {
    Write-Output 'Removed the block.'
}
