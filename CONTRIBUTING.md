# Contributing

Thanks for taking the time to contribute! This project is research-oriented and moves quickly, so lightweight contributions are welcome.

## Quick guidelines

- Keep changes focused and small.
- Prefer adding short notes in `README.md` when introducing new scripts.
- Avoid committing large datasets or checkpoints; use external storage instead.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Submitting changes

1. Create a feature branch.
2. Add or update documentation if behavior changes.
3. Ensure scripts run without errors on CPU (or MPS/CUDA if relevant).
4. Open a pull request with a concise description and screenshots (if visual output changed).
