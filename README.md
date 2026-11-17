# Transcribe Local

Offline transcription of research interviews and focus groups for MAXQDA. Recordings and
transcripts never leave the computer. It checks the computer it's installed on and adapts: it
uses an NVIDIA graphics card when there is one, sized to its memory, and otherwise runs on the
processor, which works but is much slower. Every computer uses the same model, so transcripts
are comparable wherever they were made.

> **Before you use it for research:** this is a free tool shared as is, with no warranty (see
> `LICENSE`). Automatic transcripts contain mistakes, so check them against the audio. Using it
> doesn't by itself meet any ethics board, IRB or data protection requirement: follow your own
> approved data management plan. It runs on Windows only for now.

## Rules for research data

- Keep recordings and transcripts in the data folder, outside this repo. By default that's
  `TranscriptionData` in your user folder, or `E:\TranscriptionData` if that folder exists.
  Setup can use another folder instead.
- The program refuses Google Drive and OneDrive folders.
- The program's Python code refuses internet connections while it runs. Compiled libraries aren't
  covered by that, so for a lock at the Windows level, run `scripts\firewall.ps1 on` from an
  administrator PowerShell window. Run `scripts\firewall.ps1 off` before installing updates or
  downloading models.
- This repo holds code only. `.gitignore` is an allow-list, so data can't be committed by accident.
- Claude Code sends what it reads to Anthropic. Only use it with non-sensitive sample audio.

## Setup on a Windows computer

You need Windows 11 (Windows 10 may work but hasn't been tried) and about 10 GB of free space. An NVIDIA graphics card is optional
but makes it many times faster.

1. Install the tools, then close and reopen PowerShell:
   `winget install --id Python.Python.3.12 --exact`, `winget install --id Git.Git --exact`
   and `winget install --id Gyan.FFmpeg --exact`.
2. Get the code, in a local folder that isn't synced to the cloud:
   `git clone https://github.com/Backyard-Ai/transcribe-local-public C:\Transcription_Local`.
3. From that folder, run `powershell -ExecutionPolicy Bypass -File scripts\setup.ps1`.
   Add `-DataFolder "D:\TranscriptionData"` to keep data somewhere other than the default.

Setup installs about 3 GB of Python packages and downloads the 3 GB model, which are the only
steps that use the internet. It finishes by checking the computer: which settings it chose,
whether the data folder is synced or encrypted, and a quick test run. To see that report again:
`.venv\Scripts\python -m transcribe_local.hardware --test`.

| Computer | Settings |
|---|---|
| NVIDIA card, 10 GB or more | Full speed |
| NVIDIA card, 6.5 to 10 GB | Fast |
| NVIDIA card, 3.5 to 6.5 GB | Lighter settings to fit |
| No NVIDIA card, or a smaller one | Processor only, much slower |

If the graphics card can't run the model at all, for example after a driver problem, the
program switches to the processor, says so in the window and the file's Status column, and
saves it in the job's record in the `work` folder.

**Laptops:** see [docs/laptop-setup.md](docs/laptop-setup.md) for a step-by-step checklist.
Plug in before long jobs, because the window warns you when the laptop is on battery. Keep
the lid open too, because closing it puts the laptop to sleep. A job that's interrupted
resumes from where it stopped. Graphics cards newer than the transcription library, such as
the RTX 50 series, take several minutes to start the first time after a driver install or
update.

**Encryption:** Windows 11 Home has *Device encryption* rather than BitLocker. The setup
check tells you which applies and whether it's on. See the laptop checklist for options.

To measure this computer's speed and graphics memory with a non-sensitive recording (numbers
only, and the transcript is deleted straight away):
`.venv\Scripts\python -m transcribe_local.benchmark "C:\path\to\sample.m4a"`.

The processor and a graphics card can word the same unclear passage differently. In a test
they agreed on 98.6% of words, differing only on one unclear word, so use the same computer for
a whole study if that matters. Each job records which it used. To use the processor on
purpose, set `TRANSCRIBE_LOCAL_DEVICE=cpu`.

## Use

Double-click `Transcribe Local.bat`, add recordings or a whole folder, and press Start. You can
queue several files, and press Stop at any time. Press Start again later to resume from where it stopped.

From a terminal instead:

```
.venv\Scripts\python -m transcribe_local "C:\Users\<you>\TranscriptionData\input\interview.m4a"
```

Output goes to the data folder's `output` folder, and never overwrites an earlier transcript:

- `<name>.docx` for MAXQDA: short paragraphs with timestamps that link to the audio, and passages to check highlighted
- `<name>_review.csv`: passages to check, including sound that has no transcript
- `<name>.txt`: plain text

Stopped jobs resume from the last saved segment when run again.

## Scoring against existing transcripts

Put each existing transcript as a `.txt` beside its recording, in its own folder or with the
same name. After transcribing, double-click `Compare Transcripts.bat`. It prints scores with
labels like "Recording 1" and no words or file names, so the report is safe to share. The key
linking labels to files, `comparison_key.csv`, stays with your transcripts and isn't for sharing.

Scores count differences, and an existing transcript can have errors or tidied wording of its
own. Numbers written differently, like "20" and "twenty", also count as differences.

## Removing a recording completely

The app deletes its working copy of the audio after each finished job, but a stopped or failed
job keeps it so it can resume. The raw transcript also stays in the data folder's `work` folder. To
remove a recording completely, delete its folder there, plus its files in `input` and `output`,
then empty the Recycle Bin. Deleting from an SSD isn't a guaranteed secure erase, so keep the
drive encrypted.

## Importing into MAXQDA

Use Import > Transcripts > Transcript with Timestamps for any recording, then choose the
matching audio file when MAXQDA asks.

Transcripts have no speaker labels, because automatic labels can be wrong. To code speakers
in MAXQDA, start each contribution with `Name:` and use Import > Convert Text to Focus Group
Transcript.

## Saving a copy to Google Drive

```
powershell -ExecutionPolicy Bypass -File scripts\save_to_drive.ps1 -Destination "G:\My Drive\Backups"
```

This saves a zip of the committed code only, with no data, models, or Git history. GitHub
stays the master copy, so edits to a copy on Drive won't flow back.

## Updating

From the program folder, run `git pull`, then run `scripts\setup.ps1` again. If the firewall
lock is on, turn it off first with `scripts\firewall.ps1 off` and back on afterwards.

## Tests

```
.venv\Scripts\python -m pytest
```

The same tests run on GitHub for every change.

## Problems, security and license

- Report bugs as a GitHub issue. **Never include real recordings, transcripts, file names, or
  screenshots that show them.** Use non-sensitive examples.
- Report security problems privately, as described in [SECURITY.md](SECURITY.md).
- To suggest a change, see [CONTRIBUTING.md](CONTRIBUTING.md).
- Download a fixed version from the [Releases](https://github.com/Backyard-Ai/transcribe-local-public/releases)
  page, or get the latest with `git pull`.
- MIT License (see [LICENSE](LICENSE)). The model and other components have their own licenses,
  listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
