# Setting up the laptop

Written for one particular laptop, but the steps apply to most Windows laptops with an NVIDIA
graphics card.

A checklist for the laptop: Windows 11 Home, Intel i7-14650HX, 16 GB memory, and an NVIDIA
GeForce RTX 5060 Laptop GPU with 8 GB of graphics memory. It's the same program as on the
desktop. It notices it's on the laptop and adjusts. Steps marked **You** are yours to do.
The rest you can do with Claude Code on the laptop, opened in the program's folder.

## 1. Before you start

- **You:** Plug the laptop in. On battery the graphics card runs well below full power, and
  the window shows a warning.
- **You:** In the NVIDIA App, turn **WhisperMode** off. It quiets the fans by limiting the
  graphics card, which slows transcription.
- **You:** The program stops the laptop from going to sleep while it works, but closing the
  lid still puts it to sleep. Either keep the lid open during long jobs, or set *Control
  Panel > Power Options > Choose what closing the lid does* to *Do nothing* when plugged in.
  A job that's interrupted resumes from where it stopped, so nothing is lost either way.
- The NVIDIA Studio driver 616.92 already on the laptop is new enough.

## 2. Install

Follow "Setup on a Windows computer" in `README.md`. To skip the 3 GB model download, you
can copy the `models` folder from the desktop's `C:\Transcription_Local` into the laptop's
copy with a USB stick. The model is public and holds no research data.

The laptop has no E: drive, so research data goes in `TranscriptionData` in your user folder
(for example `C:\Users\<you>\TranscriptionData`). Setup creates it.

## 3. Read the setup report

Setup ends by checking the laptop. What to expect:

- **Settings:** the RTX 5060 with 8 GB of memory and the word *fast*. That's the "medium
  card" settings: the same model as the desktop, working through half as much audio at once.
- **A note that the card is newer than the transcription library.** The library (CTranslate2)
  doesn't have ready-made code for RTX 50 series cards, so the NVIDIA driver translates it the
  first time. The quick test may sit for several minutes. That's expected once, and again
  after each driver update. Later runs start normally.
- **Quick test: passed.** If it says FAILED and mentions the graphics card, update the NVIDIA
  driver and run the check again (`.venv\Scripts\python -m transcribe_local.hardware --test`).
  Until it passes, the program still works, but it falls back to the much slower processor
  and says so in the window.
- **Drive encryption.** See the next step.

## 4. Encryption (Windows 11 Home)

Windows 11 Home has no BitLocker settings. It has **Device encryption** instead, which is
BitLocker underneath, but only on some laptops. The setup report shows the state:

| Report says | Meaning |
|---|---|
| on | Protected. Nothing to do. |
| WAITING FOR ACTIVATION | Common on new laptops. Not protected yet: turn it on as below. |
| OFF or not available | Not protected. |
| SUSPENDED | Paused, usually for an update. Restart the laptop and check again. |
| couldn't check, or unknown | Check it yourself in Settings, then run the check again. |

Choose one of these. It's your decision, and worth checking against your IRB data plan:

1. **Device encryption** (free). *Settings > Privacy & security > Device encryption*, turn
   it on. You must be signed in with a Microsoft account, and the recovery key is saved to
   that account. If the page isn't there, this laptop can't use it.
2. **Upgrade to Windows 11 Pro** (paid). This gives full BitLocker, and you choose where the
   recovery key is kept, for example printed or on a USB stick. Choose this if the key must
   stay off Microsoft's servers.
3. **VeraCrypt** (free, open source). An encrypted container just for `TranscriptionData`.
   It means more steps each time: you unlock it before transcribing.

Run the setup check again afterwards to confirm it reports *on*.

## 5. Test with non-sensitive audio

Use audio that holds nothing confidential, such as a few minutes of yourself reading a
book aloud. Use the same file on both computers.

1. **Speed and memory.** Run:

   ```
   .venv\Scripts\python -m transcribe_local.benchmark "C:\path\to\sample.m4a"
   ```

   It prints numbers only: speed, which settings were used, and graphics memory. Its
   transcript is deleted straight away. Run it twice: the first run includes the one-time
   driver translation. On the desktop, the graphics card ran at about 45x real time on a
   5-minute clip.
2. **Same words on both computers.** Transcribe the sample on the desktop as usual. Copy the
   desktop's `.txt` transcript of it into the laptop's `TranscriptionData\input`, next to the
   same sample recording and with the same name. Transcribe it on the laptop. Then
   double-click `Compare Transcripts.bat`. Near 100% means the two computers agree.

Share the benchmark printout and the comparison score with Claude so the settings can be
adjusted if needed. Both contain numbers only.

## 6. Lock it down

- **You:** From an administrator PowerShell window in the program folder, run
  `scripts\firewall.ps1 on`. This stops the program's Python from reaching the internet at
  the Windows level, on top of the program's own block.
- **You:** Make sure `TranscriptionData` isn't inside OneDrive. The program refuses OneDrive
  and Google Drive folders, and the default location is outside them.

Only then add real recordings.
