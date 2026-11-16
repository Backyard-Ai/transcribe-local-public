# Security

Transcribe Local is used with confidential research recordings, so problems that could leak
data matter most. Examples: anything that sends audio or text over the internet, writes data
into a synced cloud folder, leaves copies behind that the README says are deleted, or lets data
be committed to Git.

## Reporting a problem

Please report security problems privately, not in a public issue. On GitHub, open the
repository's **Security** tab and choose **Report a vulnerability**. You'll get a reply as soon
as possible, usually within a week.

**Never attach real recordings, transcripts, file names or screenshots that show them**, in a
report or anywhere else. Describe the problem with made-up or non-sensitive examples.

## What the program does to protect data

- Transcription runs entirely on your computer. The only internet step is the one-time model
  download during setup, and it fetches one fixed version of the model.
- While it runs, the program's Python code refuses internet connections
  (`safety.block_network`). For a Windows-level block that also covers compiled libraries, run
  `scripts\firewall.ps1 on` from an administrator PowerShell window.
- It refuses to read or write data in Google Drive or OneDrive folders.
- `.gitignore` is an allow-list, and it also blocks audio, transcript and spreadsheet file types,
  so data can't be committed by accident.
- Working copies of the audio are deleted after each finished job.

## What it can't protect against

- An unencrypted drive. Turn on BitLocker or Device encryption, or use an encrypted container.
- Other programs on the same computer, including cloud backup software that covers your data
  folder.
- Mistakes in the transcript. Always check transcripts against the audio before relying on them.
- AI coding assistants. Tools like Claude Code send what they read to their provider, so never
  point them at real recordings or transcripts.

Using this program doesn't by itself meet any ethics board, IRB or data protection
requirement. Check your own approved data management plan.

## Supported versions

Only the latest version on the `main` branch gets fixes.
