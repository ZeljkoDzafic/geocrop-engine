# Architecture Decision Records

Short records of decisions that shape GeoCrop Engine's structure. One file per decision, numbered, never
rewritten after acceptance: a changed decision gets a new ADR that supersedes the old one.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-layered-architecture.md) | Layered architecture enforced by tests | Accepted |
| [0002](0002-plugin-entry-points.md) | Plugins through Python entry points | Accepted |
| [0003](0003-file-contracts-and-schema-versions.md) | File contracts with schema versions | Accepted |
| [0004](0004-single-machine-execution.md) | Single-machine execution | Accepted |

## Template

Copy into `NNNN-short-title.md`:

```markdown
# NNNN. Title

* Status: Proposed | Accepted | Superseded by NNNN
* Date: YYYY-MM-DD

## Context
What forces are at play (scientific, technical, operational)? What problem needs a decision?

## Decision
What we do, stated in full sentences. Include the rule and where it is enforced.

## Consequences
What becomes easier, what becomes harder, what we accept. How to revisit the decision.
```
