# Genesis Cloud Deployment Guide

This directory contains deployment configurations for running Genesis in cloud environments.

## Quick Start

### Docker

Build and run locally:

```bash
# Build the image
docker build -t genesis:latest .

# Run the API server
docker run -p 8000:8000 genesis:latest

# Test the API
curl http://localhost:8000/health
```

### Docker Compose

For a complete setup with Redis for async jobs:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f genesis-api

# Stop services
docker-compose down
```

### Kubernetes with Helm

```bash
# Add dependencies (if using Redis chart)
cd deployment/helm/genesis
helm dependency update

# Install
helm install genesis ./deployment/helm/genesis \
  --namespace genesis \
  --create-namespace

# Upgrade
helm upgrade genesis ./deployment/helm/genesis -n genesis

# Uninstall
helm uninstall genesis -n genesis
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GENESIS_LOG_LEVEL` | Logging level | `INFO` |
| `GENESIS_MAX_WORKERS` | Max worker processes | `4` |
| `OPENAI_API_KEY` | OpenAI API key for LLM features | - |
| `REDIS_URL` | Redis URL for async jobs | - |

### Helm Values

Key configuration options in `values.yaml`:

```yaml
# Scaling
replicaCount: 2
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10

# Resources
resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 1Gi

# Persistence for models
persistence:
  enabled: true
  size: 10Gi

# Enable Redis for async jobs
redis:
  enabled: true
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/generate` | POST | Generate synthetic data |
| `/v1/evaluate` | POST | Evaluate data quality |
| `/v1/models` | GET | List trained models |
| `/v1/jobs/{id}` | GET | Check async job status |

## Production Recommendations

1. **Enable TLS**: Configure Ingress with TLS termination
2. **Resource limits**: Set appropriate CPU/memory limits
3. **Persistence**: Enable PVC for model storage
4. **Autoscaling**: Enable HPA for production workloads
5. **Monitoring**: Enable metrics and set up Prometheus/Grafana
6. **Security**: Use network policies and pod security contexts

## Serverless Deployment

### AWS Lambda

Use the `genesis.api.lambda_handler` for AWS Lambda deployment:

```python
from genesis.api.lambda_handler import handler

def lambda_handler(event, context):
    return handler(event, context)
```

### Google Cloud Run

```bash
# Build for Cloud Run
gcloud builds submit --tag gcr.io/PROJECT/genesis

# Deploy
gcloud run deploy genesis \
  --image gcr.io/PROJECT/genesis \
  --platform managed \
  --memory 4Gi \
  --cpu 2
```
