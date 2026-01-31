# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.4.x   | :white_check_mark: |
| 1.3.x   | :white_check_mark: |
| < 1.3   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

**Please DO NOT open a public GitHub issue for security vulnerabilities.**

Instead, report vulnerabilities via email to: **security@genesis-synth.io**

Include the following information:

1. **Description**: A clear description of the vulnerability
2. **Steps to Reproduce**: Detailed steps to reproduce the issue
3. **Impact**: Potential impact of the vulnerability
4. **Affected Versions**: Which versions are affected
5. **Suggested Fix**: If you have one (optional)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt within 48 hours
- **Initial Assessment**: Within 7 days, we will provide an initial assessment
- **Resolution Timeline**: We aim to resolve critical issues within 30 days
- **Credit**: We will credit reporters in security advisories (unless you prefer anonymity)

### Scope

The following are in scope for security reports:

- Genesis library code (`genesis/` directory)
- Official Docker images
- Documentation that could lead to insecure configurations

The following are out of scope:

- Third-party dependencies (report to upstream maintainers)
- Social engineering attacks
- Denial of service attacks

## Security Best Practices

When using Genesis:

1. **API Keys**: Never commit API keys. Use environment variables.
2. **Privacy Settings**: Enable differential privacy for sensitive data.
3. **Network**: Use TLS when deploying the REST API in production.
4. **CORS**: Configure specific allowed origins instead of wildcards in production.

## Past Security Advisories

No security advisories have been published yet.
