# Assets

Visual assets used by the README and `frontend/`.

## Inventory (planned)

| File | Purpose | Captured during |
|---|---|---|
| `screenshot-dashboard.png` | Gradio dashboard with histogram + per-episode dataframe | AWS validation run |
| `screenshot-rationale.png` | Close-up of one critic card with rationale + timestamp evidence | AWS validation run |
| `architecture.png` | Rendered version of the ASCII architecture diagram | manual export from `docs/architecture.md` |
| `demo.gif` | 10-second screencap: paste URL → score → threshold → push | best-effort during AWS validation |

## How to add a new asset

1. Drop the file in this directory.
2. Reference it from `README.md` with a relative path: `![alt](assets/foo.png)`.
3. Keep file sizes reasonable: aim for <500 KB per PNG, <2 MB per GIF.
   Larger files are fine in this repo (we're not size-constrained), but
   lighter images render faster in the GitHub README preview.
4. Don't commit screenshots that include API keys, HF tokens, or other
   secrets visible in the dev tools / terminal.

## License

All assets in this directory are released under the same MIT license as
the rest of the project unless explicitly noted otherwise.
