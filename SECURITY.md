# Security policy

## Supported versions

Crucible is pre-1.0. We patch security issues on the latest minor line
only. Older versions do not receive security updates.

| Version | Supported |
|---|---|
| 0.1.x   | Yes |
| < 0.1.0 | No  |

## Reporting a vulnerability

If you find a security issue, please **do not** open a public GitHub
issue. Report it privately:

1. Open a [GitHub Security Advisory](https://github.com/lord-arbiter/Crucible/security/advisories/new)
   on the repo. Maintainers receive these privately.
2. Or email the maintainer listed in the GitHub repo profile.

We aim to acknowledge reports within 72 hours and ship a fix or
mitigation within 14 days for high-severity issues, longer for
lower-severity ones. Please don't disclose publicly until a fix is
released.

## Known sensitive surfaces

When auditing a Crucible deployment, pay special attention to:

- **HuggingFace tokens.** The push-to-Hub flow accepts a write-scoped
  HF token via the Gradio UI or the `/push_filtered` API. Tokens are
  used in-process and not persisted, but a compromised orchestrator
  could exfiltrate them. Run the orchestrator on a trusted host.
- **VLM API keys.** `CRUCIBLE_VLM_API_KEY` is read from environment.
  Don't commit it to the repo. The Space loads it from secrets, not
  from `.env`.
- **CORS.** `src/api.py` opens CORS to `*` so the HF Space can reach the
  GPU box directly. If you deploy the FastAPI orchestrator on a public
  host, scope `allow_origins` to your Space URL only.
- **Input validation.** Crucible accepts arbitrary HuggingFace dataset
  repo IDs. We don't sandbox the dataset reader — a malicious dataset
  could ship malformed parquet that triggers PyArrow vulnerabilities.
  Stick to `lerobot/*` and other vetted publishers, or run the
  orchestrator in a network-isolated container.
- **Model output trust.** Critic rationale strings are rendered as HTML
  in the Gradio frontend. We escape via Gradio's HTML component
  defaults, but if you pipe outputs into a different renderer be aware
  that VLM outputs are not inherently trustworthy.

## Dependencies

We pin major versions in `pyproject.toml` and CI runs against the
latest minor in each major. Critical CVEs in pinned deps will be
patched on the latest 0.1.x line within 14 days of public disclosure.
