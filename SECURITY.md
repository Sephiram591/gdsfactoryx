# Security Policy

## Supported Versions

| Version                | Supported             |
| ---------------------- | --------------------- |
| 9.51.x (`gdsfactoryx`) | Latest release only   |
| < 9.51                 | :x:                   |

Only the latest release receives security updates.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them privately through [GitHub Security Advisories](https://github.com/Sephiram591/gdsfactoryx/security/advisories).

Include as much of the following as possible:

- Description of the vulnerability
- Steps to reproduce or a proof of concept
- Affected version(s)
- Potential impact

We will acknowledge receipt, work with you to understand the issue, and coordinate a timeline for a fix.

## Disclosure Policy

- We ask that you give us reasonable time to address the issue before any public disclosure.
- We will coordinate with you on the disclosure timeline and credit you (unless you prefer to remain anonymous).
- Once a fix is released, we will publish a security advisory on GitHub.

## Scope

This policy applies to the `gdsfactoryx` Python package (providing the `gdsfactory` module) maintained at [Sephiram591/gdsfactoryx](https://github.com/Sephiram591/gdsfactoryx). For vulnerabilities specific to upstream `gdsfactory`, see [gdsfactory/gdsfactory](https://github.com/gdsfactory/gdsfactory/security).

## Security Best Practices for Users

- Pin your dependencies to specific versions in production environments.
- Keep `gdsfactoryx` updated to the latest supported version.
- Be cautious when loading GDS files or cell definitions from untrusted sources, as they may execute arbitrary Python code.
