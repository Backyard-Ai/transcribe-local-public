<#
Sets up Transcribe Local on a Windows computer, or updates an existing setup.

Run from the repo folder:
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -DataFolder "D:\TranscriptionData"

It installs the Python packages, downloads the model (the only step that needs the
internet), creates the data folder, then checks the computer and picks settings to suit it.
Without -DataFolder, data goes in E:\TranscriptionData if that folder exists, otherwise in
TranscriptionData inside your user folder. Running it again is safe.
#>
param([string]$DataFolder)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')

function Find-Python {
    # The "python" Windows ships with is only a shortcut to the Microsoft Store, so test each one.
    # A missing version writes to stderr, which mustn't stop the script.
    $ErrorActionPreference = 'Continue'
    foreach ($candidate in @(@('py', '-3.12'), @('py', '-3'), @('python'))) {
        if (-not (Get-Command $candidate[0] -ErrorAction SilentlyContinue)) { continue }
        $arguments = @($candidate | Select-Object -Skip 1) + @('-c', 'import sys; print(sys.version_info >= (3, 11))')
        $answer = & $candidate[0] @arguments 2>$null
        if ($LASTEXITCODE -eq 0 -and $answer -eq 'True') { return , $candidate }
    }
    return $null
}

function Step($text) { Write-Output ''; Write-Output "== $text" }

Step 'Checking the tools this needs'
$python = Find-Python
$missing = @()
if (-not $python) { $missing += 'winget install --id Python.Python.3.12 --exact' }
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { $missing += 'winget install --id Gyan.FFmpeg --exact' }
if ($missing) {
    Write-Output 'Install these first, then close and reopen PowerShell and run this again:'
    $missing | ForEach-Object { Write-Output "  $_" }
    exit 1
}
Write-Output 'Python and ffmpeg found.'

$venvPython = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Step 'Creating the Python environment'
    $arguments = @($python | Select-Object -Skip 1) + @('-m', 'venv', (Join-Path $root '.venv'))
    & $python[0] @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Creating the Python environment failed.' }
}

Step 'Installing Python packages (about 3 GB the first time)'
& $venvPython -m pip install --upgrade pip --quiet
& $venvPython -m pip install -r (Join-Path $root 'requirements.txt') --quiet
if ($LASTEXITCODE -ne 0) { throw 'Installing packages failed.' }
& $venvPython -m pip install -e $root --no-deps --quiet
if ($LASTEXITCODE -ne 0) { throw 'Installing the program failed.' }

if (-not (Test-Path (Join-Path $root 'models\faster-whisper-large-v3\model.bin'))) {
    Step 'Downloading the model (about 3 GB)'
    & $venvPython (Join-Path $root 'scripts\download_models.py')
    if ($LASTEXITCODE -ne 0) { throw 'Downloading the model failed.' }
}

if ($DataFolder) {
    Step 'Saving the data folder setting'
    $settings = @{ data_folder = [IO.Path]::GetFullPath($DataFolder) } | ConvertTo-Json
    [IO.File]::WriteAllText((Join-Path $root 'local_settings.json'), $settings)
}

Step 'Creating the data folder'
$dataRoot = & $venvPython -c 'from transcribe_local.pipeline import DATA_ROOT; print(DATA_ROOT)'
foreach ($name in 'input', 'output', 'work') { New-Item -ItemType Directory -Force (Join-Path $dataRoot $name) | Out-Null }
Write-Output "Data folder: $dataRoot"

Step 'Checking this computer'
& $venvPython -m transcribe_local.hardware --test
$checkResult = $LASTEXITCODE
Write-Output ''
if ($checkResult -eq 0) {
    Write-Output 'Setup finished. Open the app with "Transcribe Local.bat".'
} else {
    Write-Output 'Setup finished, but the check above found a problem to fix first.'
}
exit $checkResult
