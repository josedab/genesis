# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for the Genesis Synthetic Data Platform.

## What are ADRs?

ADRs document significant architectural decisions made during the development of this project. They capture the context, decision, and consequences of each choice, helping new team members understand "why is it built this way?"

## ADR Index

### Core Architecture (v1.0)

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-0001](0001-sklearn-style-fit-generate-api.md) | Sklearn-Style Fit/Generate API Pattern | Accepted | 2024-01 |
| [ADR-0002](0002-multi-backend-deep-learning-support.md) | Multi-Backend Deep Learning Support | Accepted | 2024-01 |
| [ADR-0003](0003-optional-dependencies-via-extras.md) | Optional Dependencies via Extras | Accepted | 2024-01 |
| [ADR-0004](0004-rejection-sampling-conditional-generation.md) | Rejection Sampling for Conditional Generation | Accepted | 2024-06 |
| [ADR-0005](0005-dataclass-based-configuration.md) | Dataclass-Based Configuration | Accepted | 2024-01 |
| [ADR-0006](0006-privacy-first-class-concern.md) | Privacy as a First-Class Concern | Accepted | 2024-01 |
| [ADR-0007](0007-gaussian-copula-fallback-generator.md) | Gaussian Copula as Fallback Generator | Accepted | 2024-01 |
| [ADR-0008](0008-statistic-aggregation-federated-learning.md) | Statistic Aggregation for Federated Learning | Accepted | 2024-06 |
| [ADR-0009](0009-layered-evaluation-architecture.md) | Layered Evaluation Architecture | Accepted | 2024-01 |
| [ADR-0010](0010-rest-api-optional-deployment.md) | REST API as Optional Deployment Mode | Accepted | 2024-06 |

### Extension & Integration (v1.2+)

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-0011](0011-plugin-architecture-decorator-registration.md) | Plugin Architecture with Decorator-Based Registration | Accepted | 2024-06 |
| [ADR-0012](0012-pandas-primary-data-interchange.md) | Pandas as Primary Data Interchange Format | Accepted | 2024-01 |
| [ADR-0013](0013-blockchain-style-lineage-tracking.md) | Blockchain-Style Immutable Lineage Tracking | Accepted | 2024-06 |
| [ADR-0014](0014-streaming-generation-kafka-websocket.md) | Streaming Generation with Kafka/WebSocket Integration | Accepted | 2024-06 |

### Intelligence & Automation (v1.4+)

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-0015](0015-automl-method-selection-meta-features.md) | AutoML Method Selection via Meta-Feature Analysis | Accepted | 2024-09 |
| [ADR-0016](0016-domain-specific-generators.md) | Domain-Specific Generators as Specialization Layer | Accepted | 2024-09 |
| [ADR-0017](0017-git-like-dataset-versioning.md) | Git-Like Dataset Versioning System | Accepted | 2024-09 |
| [ADR-0018](0018-compliance-certificates-first-class.md) | Compliance Certificates as First-Class Artifacts | Accepted | 2024-06 |
| [ADR-0019](0019-constraint-system-post-generation.md) | Constraint System with Post-Generation Validation | Accepted | 2024-01 |
| [ADR-0020](0020-cli-thin-wrapper-over-sdk.md) | CLI as Thin Wrapper Over Python SDK | Accepted | 2024-01 |

## ADR Template

When creating a new ADR, use this template:

```markdown
# ADR-NNNN: Title

## Status
Accepted | Superseded | Deprecated

## Context
What prompted this decision?

## Decision
What was decided?

## Consequences
What are the tradeoffs and implications?
```

## References

- [Michael Nygard's original ADR article](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [ADR GitHub Organization](https://adr.github.io/)
