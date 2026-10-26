# Security policy

## Supported versions

| Version | Security fixes |
|---|---|
| latest minor release (currently 0.9.x) | yes |
| older releases | no — upgrade to the latest release |

See [docs/release.md](docs/release.md) for the support matrix and versioning policy.

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Report them privately via
[GitHub private vulnerability reporting](https://github.com/ZeljkoDzafic/geocrop-engine/security/advisories/new)
or by e-mail to zeljkodzafic@gmail.com with the subject "GeoCrop security".

Include the affected version or commit, the component (for example the local web API, a CLI command or a file
parser), steps to reproduce and the impact you expect. You will receive an acknowledgement within 7 days and an
assessment within 30 days. Fixes are released as a patch version with a CHANGELOG entry and, where appropriate,
a GitHub security advisory crediting the reporter (unless you prefer to remain anonymous).

## Scope

GeoCrop Engine's optional web workspace is a single-owner tool bound to the loopback interface. Reports are in
scope when they let a party **without the access token** — for example a website open in the owner's browser or
another local account — read data, run commands, or bypass the documented restrictions (blocked commands, output
confinement, artifact download confinement). Behaviour that requires the token, or that follows from loading
untrusted inputs or `joblib` model files, is documented as out of scope in the
[security model](docs/security.md).
