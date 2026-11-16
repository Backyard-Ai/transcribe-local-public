<#
Saves a clean zip of the committed code to Google Drive, for your records.

Only committed files go in the zip, so it can't include recordings, transcripts,
models, Git history, or the Python environment. GitHub stays the master copy:
edits made to a copy on Drive won't flow back.

Usage:
  scripts\save_to_drive.ps1 -Destination "G:\My Drive\Some Folder"
#>
param([Parameter(Mandatory)][string]$Destination)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$git = (Get-Command git -ErrorAction SilentlyContinue).Source
if (-not $git) { $git = 'C:\Program Files\Git\cmd\git.exe' }

if (& $git -C $root status --porcelain) {
    throw 'There are uncommitted changes, and they would be missing from the zip. Commit them first.'
}
if (-not (Test-Path -LiteralPath $Destination)) {
    throw "The folder $Destination doesn't exist."
}

$version = (Select-String -LiteralPath (Join-Path $root 'pyproject.toml') -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value
$commit = & $git -C $root rev-parse --short HEAD
$zip = Join-Path $Destination "transcribe_local_v${version}_$commit.zip"
if (Test-Path -LiteralPath $zip) {
    Write-Output "Already saved: $zip"
    return
}

& $git -C $root archive --format=zip --prefix=transcribe_local/ -o $zip HEAD
if ($LASTEXITCODE -ne 0) { throw 'git archive failed.' }
Write-Output "Saved $zip"
