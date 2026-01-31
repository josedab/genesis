# Architecture Documentation

This directory contains architecture documentation for Genesis.

## Contents

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Comprehensive architecture overview with diagrams
- **[overview.md](overview.md)** - High-level system architecture
- **[v130_features.md](v130_features.md)** - v1.3.0 architecture additions
- **[v140_features.md](v140_features.md)** - v1.4.0 architecture additions
- **[decisions/](decisions/)** - Architecture Decision Records (ADRs)

---

# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for Genesis.

## What is an ADR?

An ADR is a document that captures an important architectural decision made along with its context and consequences.

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [001](decisions/001-framework-agnostic-backends.md) | Framework-Agnostic Deep Learning Backends | Accepted | 2024-01 |
| [002](decisions/002-sklearn-style-api.md) | Sklearn-Style fit/generate API | Accepted | 2024-01 |
| [003](decisions/003-constraint-system.md) | Declarative Constraint System | Accepted | 2024-01 |
| [004](decisions/004-privacy-by-design.md) | Privacy-by-Design Architecture | Accepted | 2024-01 |

## Creating a New ADR

1. Copy the template below
2. Create a new file: `docs/architecture/decisions/NNN-title.md`
3. Fill in the sections
4. Add to the index above

## Template

```markdown
# ADR NNN: Title

## Status

Proposed | Accepted | Deprecated | Superseded by [ADR XXX](XXX-title.md)

## Context

What is the issue that we're seeing that is motivating this decision or change?

## Decision

What is the change that we're proposing and/or doing?

## Consequences

What becomes easier or more difficult to do because of this change?

### Positive

- Benefit 1
- Benefit 2

### Negative

- Drawback 1
- Drawback 2

### Neutral

- Note 1
```
