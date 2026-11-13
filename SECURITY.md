# Security Policy

Do not open a public issue for a suspected credential, privacy exposure, or exploitable flaw.
Use GitHub's private security-advisory interface for this repository. Include the affected path,
impact, and reproduction steps, but do not include live credentials.

Supported release: `1.x`. Model credentials are read from environment variables and must never be
stored in configuration, logs, trajectories, or issue attachments. Rotate any exposed credential
before reporting it.
