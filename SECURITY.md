# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately using a GitHub private
security advisory for the Official Repository. Do not disclose vulnerabilities,
credentials, tokens, private keys, or exploit details in public issues,
discussions, pull requests, or commits.

Include a concise description, affected version or commit, reproducible steps,
potential impact, and any suggested mitigation. The project maintainer will
review reports as capacity permits and may coordinate a fix and disclosure.

## Sensitive information

Do not commit API keys, passwords, access tokens, SSH keys, `.env` files,
database or cloud credentials, service-account files, private models, or other
confidential material. If a secret is accidentally committed, revoke or rotate
it immediately with its provider, remove it from the current code and Git
history as appropriate, and report the exposure privately. Removing a file in a
later commit does not invalidate an exposed credential.

The repository's `.gitignore` blocks common local credential files, but it is
not a substitute for review or GitHub secret scanning.