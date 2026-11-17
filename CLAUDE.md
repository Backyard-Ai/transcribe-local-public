# Notes for Claude

Offline transcription of research interviews and focus groups for MAXQDA. Built and tested on
the owner's desktop (Windows 11, Ryzen 9 7900X, 64 GB RAM, RTX 3060 12 GB). The owner is a
qualitative researcher, not a developer: explain in plain language and avoid jargon.

## Privacy rules (most important)

- Recordings and transcripts are confidential research data. Never read, open, list by name,
  or print the contents of real recordings or transcripts. Anything you read is sent to
  Anthropic. Work with non-sensitive sample audio or made-up test data only.
- Real data lives in the data folder (`input`, `output`, `work`): `E:\TranscriptionData` on the
  desktop, `TranscriptionData` in the user folder elsewhere. When you need to check that folder,
  report counts, types and sizes only, never file names.
- Never put research data on Google Drive, OneDrive or GitHub. `safety.py` refuses synced
  folders and `.gitignore` is an allow-list; keep both that way.
- The program's Python code blocks network connections while it runs (`safety.block_network`).
  The only network step is the one-time model download.
- Ask before pushing to GitHub. The owner works in the private repo `Backyard-Ai/transcribe-local`
  and shares a public copy, `Backyard-Ai/transcribe-local-public`, with its own separate history.
  Never add the public repo as a remote of the private one. To publish, copy the files of a
  committed version into a clone of the public repo and commit there with the owner's GitHub
  no-reply address. Treat everything as if it will be public: no user names, email addresses,
  personal folder paths, or anything about real recordings in code, docs, tests or commit
  messages. Other people may use this repo with Claude Code, and these rules protect their data too.
- Changing firewall or encryption settings is for the owner to do; give them the steps.

## How it works

`gui.py` (window) and `cli.py` (command line) both call `pipeline.process`, which:
converts audio with ffmpeg to 16 kHz mono WAV in `work/<name>-<hash>/`, transcribes with
faster-whisper large-v3 on CUDA (`engine.py`), saves each segment to `segments.jsonl` as it
finishes so jobs resume after a crash, writes DOCX, TXT and a review CSV, then deletes the
working WAV. `compare.py` scores the app's transcripts against existing ones and prints
numbers and anonymous labels only.

On the card, CTranslate2 uses NVIDIA's cuBLAS and cuDNN from CUDA 12, installed from NVIDIA's
own packages and made findable by `cuda_libraries.py`. `hardware.py` reads the card's name,
memory and generation from `nvidia-smi`, which comes with every NVIDIA driver.

## Decisions already made with the owner, don't revisit without asking

- No speaker labels. Automatic labels were wrong in testing, and a wrong label is worse than none.
- DOCX for MAXQDA: paragraphs break at pauses of 1.5 s or after 20 s, each ending in a
  `[h:mm:ss.x]` timestamp. Import with Import > Transcripts > Transcript with Timestamps.
  The owner confirmed the import, timestamps and yellow highlighting work in the latest MAXQDA.
- Passages to check are highlighted yellow and listed in `_review.csv`. Word confidence is
  judged per sentence. The thresholds in `review.py` are untuned starting values.
- The voice detector threshold is 0.1, not the default 0.5, because the default silently
  dropped real speech in quiet passages. `coverage.py` separately lists loud audio with no
  transcript, so dropped speech can't go unnoticed.
- English only. The working WAV is deleted after a finished job; stopped or failed jobs keep
  it so they can resume. Transcripts never overwrite earlier ones.

## Running it on another computer (the laptop)

One program adapts to each computer (the owner's choice, not separate copies). `hardware.py`
detects the computer and picks a profile: NVIDIA cards by memory size (10 GB+, 6.5 GB+, 3.5 GB+,
each ending with the processor as a last resort), otherwise the processor. The desktop's 12 GB
card keeps its original settings exactly. All computers use the same model on purpose, so
transcripts are comparable; only device, compute type and batch size change, and each job
records which it used. Changing `engine.SETTINGS` starts fresh transcripts, by design.

To set up a new Windows computer, follow `README.md`: install the tools, clone, then run
`scripts\setup.ps1`. It installs packages, downloads the model, creates the data folder
(default `TranscriptionData` in the user folder if `E:\TranscriptionData` doesn't exist, or
`-DataFolder` saved in `local_settings.json`, which Git ignores), then runs
`python -m transcribe_local.hardware --test`. Read that report with the owner. Before real
recordings, test with non-sensitive audio and remind the owner to turn on BitLocker.

Measured on the desktop with a 5-minute clip: graphics card 45x real time, its 12-core
processor 6.6x. A laptop processor will usually be slower than that. The 6.5 GB and 3.5 GB card
boundaries are estimates not yet tried on real cards of those sizes. Mac isn't supported yet:
the setup script, launchers and firewall script are Windows-only.

### The owner's laptop

Windows 11 Home, i7-14650HX (24 threads), 16 GB memory, RTX 5060 Laptop GPU with 8 GB
(8151 MB, so the medium card profile: float16 batch 4 first). Follow `docs/laptop-setup.md`
with the owner, step by step.

- The RTX 5060 is a Blackwell card (generation 12.0). The CTranslate2 4.8.5 and 4.9.0 Windows
  wheels hold ready-made code only up to generation 8.6, plus portable code (PTX) that the
  driver translates on first use. So it should work, with a slow first start after each driver
  install. This was checked by reading the wheel, not on the card. `hardware.LIBRARY_NEWEST_CARD`
  drives the first-run note. `__init__.py` raises `CUDA_CACHE_MAXSIZE` so the translation is kept.
- If the card can't run the model (an error naming CUDA, cuDNN, cuBLAS or a DLL that isn't
  out-of-memory), `engine.transcribe` skips the remaining card attempts, continues on the
  processor from the last saved segment, tells the window, and records `card_problem` in the
  job. Later jobs in the same session skip the card. This was the recommended option in the
  laptop plan. The owner may still choose "stop and tell me" instead.
- On battery, the window and the setup report show a warning (`hardware.on_battery`).
- Windows Home: the encryption check reports Device encryption, including "waiting for
  activation" (encrypted but not protected). Which encryption to use (Device encryption, Pro
  with BitLocker, or VeraCrypt) is the owner's decision. Don't choose for them.
- `python -m transcribe_local.benchmark <non-sensitive sample>` measures speed and peak
  graphics memory (via nvidia-smi), prints numbers only, and deletes its transcript. After the
  laptop run, record its numbers here and adjust `MEDIUM_CARD_GB` only if they call for it.

### Laptop results (2026-11-15)

- Setup: Python 3.12.10, Git, GitHub CLI and FFmpeg 9.0.2 installed with winget, then
  `scripts\setup.ps1` ran cleanly. Data folder: the default `TranscriptionData` in the user
  folder, local.
- Setup report: medium card profile ("RTX 5060 Laptop GPU (8 GB): fast"). The quick test passed
  on the card in 32 s, including the first-run driver translation, far less than the several
  minutes the note warns of.
- Encryption: the report showed Device encryption already **on**. Nothing was changed.
- Real use: 7 of the owner's recordings transcribed on the card. All 7 jobs recorded CUDA and
  float16 with no `card_problem`, so the first card setting worked and no fallback happened.
  (Checked by counting fields in the job files, not by reading them.)
- The benchmark was not run: the owner chose to skip it. `MEDIUM_CARD_GB` stays as it is.
- Open decision: the owner's IRB requires recordings to stay in the university OneDrive, but
  `pipeline.process` refuses a recording in any OneDrive folder (`require_local(source, ...)`).
  Transcripts and working files are already local. Options put to the owner: allow reading
  recordings from OneDrive while still refusing OneDrive for transcripts and working files, or
  copy recordings to a local folder for each job. Not decided yet. Don't change it without asking.
- Google Drive on the laptop isn't at `G:\My Drive`. `scripts\save_to_drive.ps1` now always
  asks for `-Destination`, so no personal folder path is stored in the repo.
- Git on the laptop already had GitHub access through Git Credential Manager. `gh` is installed
  but not signed in, and nothing needs it.

## Tested and not tested

Tested on the desktop: 163 automated tests (219 since the laptop work, run on Linux only so
far, where one Windows-only test is skipped); a 6-hour looped file with a forced crash and
resume; graphics memory freed on the out-of-memory fallback; a fresh job under `pythonw`;
.m4a, .mp3, .wav, .flac, .ogg, .wma, .aac and .opus; the MAXQDA import; the comparison tool on
made-up transcripts; the setup script; and real transcription forced onto the processor.

Tested on the laptop (see "Laptop results"): the setup script, the RTX 5060 with the medium card
profile on real recordings, and the encryption check reporting Device encryption on.

Not tested: the laptop's speed (benchmark skipped), the battery warning and the card fallback on
real Windows (the laptop was plugged in and the card never failed); accuracy on the owner's real, often poor-quality recordings; a real out-of-memory
event; video files; the firewall rules themselves (they need administrator rights); the medium
and small card profiles on real cards; and any computer other than the desktop.

## Working on the code

- Tests: `.venv\Scripts\python -m pytest`. Add a test with each change.
- Run the window with `Transcribe Local.bat`, or the command line with
  `.venv\Scripts\python -m transcribe_local <file>`.
- Save a code zip to Drive with `scripts\save_to_drive.ps1 -Destination <folder>` (code only,
  never data).
- Getting ready to share (2026-11-16): MIT `LICENSE`, `SECURITY.md`, `THIRD_PARTY_NOTICES.md`
  (all dependencies and the model are permissive; FFmpeg is installed separately, not bundled),
  an issue template warning against posting data, a Windows test run on GitHub
  (`.github/workflows/tests.yml`, no model needed, ends with `pip check`) and monthly
  Dependabot updates that skip ctranslate2, faster-whisper, onnxruntime and the NVIDIA
  packages, since those change transcripts and need testing on real cards.
  `scripts/download_models.py` pins the model to one revision. New top-level files must be
  added to the `.gitignore` allow-list. The public repo's `main` is protected: no deletion or
  force-push for anyone, and pull requests need the "test" check; the owner (repository admin)
  is exempt from the check, which is what lets publishing push straight to `main`.
- PyTorch removed (0.1.1, 2026-11-16). It was installed only to make CUDA's libraries findable
  and to read the card's details, and it carried security alerts. PyTorch 2.11 is the last
  build with CUDA 12.8; newer ones are CUDA 13, which CTranslate2 4.8.5 can't use, or CUDA
  12.6, which gave identical transcripts but ran at half speed on the RTX 5060 (its libraries
  predate Blackwell). The NVIDIA packages are pinned to exactly what PyTorch 2.11 supplied
  (cuBLAS 12.8.4.1, cuDNN 9.19.0.56, CUDA runtime 12.8.90). Tested on the laptop's RTX 5060
  with a Windows text-to-speech sample: word-for-word the same transcript as with PyTorch,
  about 12.7x real time against 13.1x, the gap being a fixed ~0.4 s for loading the libraries,
  which PyTorch used to do before the timer. Installed size went from 4.5 GB to 1.9 GB.
  `setup.ps1` removes PyTorch from existing installs. Still to check on the desktop's RTX 3060:
  run `git pull` and `scripts\setup.ps1`, then confirm the quick test passes on the card.
- Commit messages end with the co-author line used in the existing history.
