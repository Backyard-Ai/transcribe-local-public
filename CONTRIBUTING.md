# Contributing

Thanks for your interest in Transcribe Local. It's a small project maintained by Backyard AI,
and changes are reviewed as time allows.

## The one rule

**Never share research data.** Don't put real recordings, transcripts, participant names, file
names or screenshots that show them in an issue, pull request, test or commit. Use made-up text
or audio that holds nothing confidential.

## Reporting a bug or asking for a feature

Open an issue and fill in the form. Paste the setup report
(`.venv\Scripts\python -m transcribe_local.hardware --test`), because it shows the computer and
settings without any recording data.

Security problems go privately through the Security tab instead. See [SECURITY.md](SECURITY.md).

## Changing the code

Please open an issue before starting anything big, so we can agree on the approach first. Some
choices are deliberate and explained in `CLAUDE.md`, such as having no speaker labels and using
the same model on every computer.

1. Fork the repo and make a branch.
2. Set up with `scripts\setup.ps1`, then make your change with a test that covers it.
3. Run the tests with `.venv\Scripts\python -m pytest`. They also run on GitHub for every pull
   request and must pass.
4. Open a pull request saying what changed and why.

Changes to the transcription libraries (PyTorch, CTranslate2, faster-whisper, onnxruntime) or the
model need testing on a real graphics card, because they can change transcripts.

By contributing, you agree that your contribution is licensed under the MIT License, like the
rest of the project.
