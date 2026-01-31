# ADR-0010: REST API as Optional Deployment Mode

## Status

Accepted

## Context

Genesis is designed primarily as a Python library for data scientists working in notebooks and scripts. However, production deployment scenarios require additional capabilities:

- **Microservice architecture**: synthetic data generation as a service
- **Language interoperability**: non-Python applications need access
- **Resource isolation**: GPU-intensive generation on dedicated servers
- **Access control**: centralized API with authentication
- **Async generation**: long-running jobs with status polling

Two architectural approaches were considered:

### Approach 1: API-First Design
Make the REST API the primary interface, with Python library as a wrapper.
- Pro: Consistent deployment model
- Con: Forces complexity on simple use cases
- Con: Latency for local operations
- Con: Heavy dependencies always required

### Approach 2: Library-First with Optional API
Design for library use, add API as a deployment layer.
- Pro: Simple for most users
- Pro: No server overhead for scripts
- Pro: API dependencies are optional
- Con: Two interfaces to maintain

## Decision

We implement the REST API as an **optional deployment mode**:

```python
# Library usage (primary)
from genesis import SyntheticGenerator
gen = SyntheticGenerator(method='ctgan')
gen.fit(data)
synthetic = gen.generate(1000)

# API usage (optional, requires genesis[api])
# Start server: uvicorn genesis.api.server:app --port 8000
```

The API is installed via an optional extra:

```bash
pip install genesis-synth[api]  # Adds fastapi, uvicorn, pydantic
```

### API Design

```python
# genesis/api/server.py
from fastapi import FastAPI

app = FastAPI(title="Genesis Synthetic Data API", version=__version__)

@app.post("/v1/generate")
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate synthetic data."""
    ...

@app.post("/v1/evaluate")
async def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    """Evaluate synthetic data quality."""
    ...

@app.get("/v1/jobs/{job_id}")
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Check status of async generation job."""
    ...
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check, uptime, version |
| POST | `/v1/generate` | Sync or async generation |
| POST | `/v1/evaluate` | Quality evaluation |
| GET | `/v1/models` | List trained models |
| GET | `/v1/jobs/{id}` | Async job status |
| DELETE | `/v1/jobs/{id}` | Cancel async job |

### Request/Response Schemas

```python
# genesis/api/schemas.py
from pydantic import BaseModel

class GenerateRequest(BaseModel):
    """Request to generate synthetic data."""
    data: List[Dict[str, Any]]  # Training data as records
    n_samples: int = 1000
    method: str = "auto"
    discrete_columns: Optional[List[str]] = None
    async_mode: bool = False  # Return job_id instead of data

class GenerateResponse(BaseModel):
    """Response with generated data."""
    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    job_id: Optional[str] = None
    generation_time: Optional[float] = None
    metadata: Dict[str, Any] = {}
```

## Consequences

### Positive

- **Simple default**: library users don't deal with API complexity
- **Lightweight install**: API dependencies only when needed
- **Clean separation**: API is a thin layer over library
- **Flexibility**: deploy as microservice or use as library
- **Standards-based**: OpenAPI/Swagger documentation auto-generated

### Negative

- **Two interfaces**: must keep API in sync with library
- **Serialization overhead**: data converted to/from JSON
- **State management**: API must manage trained models
- **Authentication**: not included, users must add

### Mitigations

1. **API wraps library**: API handlers call library functions
   ```python
   @app.post("/v1/generate")
   async def generate(request: GenerateRequest):
       gen = SyntheticGenerator(method=request.method)
       gen.fit(pd.DataFrame(request.data))
       result = gen.generate(request.n_samples)
       return GenerateResponse(data=result.to_dict('records'))
   ```

2. **Pydantic validation**: schemas validate requests before processing

3. **OpenAPI spec**: auto-generated docs at `/docs` and `/redoc`

4. **Auth middleware**: users can add FastAPI security middleware
   ```python
   from genesis.api import create_app
   app = create_app()
   app.add_middleware(AuthMiddleware, ...)
   ```

## Deployment Options

### Docker

```dockerfile
FROM python:3.10-slim
RUN pip install genesis-synth[pytorch,api]
CMD ["uvicorn", "genesis.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
services:
  genesis-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GENESIS_WORKERS=4
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

### Kubernetes (Helm)

```bash
helm install genesis ./deployment/helm/genesis \
  --set api.replicas=3 \
  --set api.resources.requests.gpu=1
```

## API Usage Examples

### Synchronous Generation

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{"age": 25, "income": 50000}, ...],
    "n_samples": 100,
    "method": "gaussian_copula"
  }'
```

### Asynchronous Generation

```bash
# Start job
curl -X POST http://localhost:8000/v1/generate \
  -d '{"data": [...], "n_samples": 10000, "async_mode": true}'
# Returns: {"job_id": "abc123", "status": "pending"}

# Poll for completion
curl http://localhost:8000/v1/jobs/abc123
# Returns: {"status": "completed", "data": [...]}
```

### Python Client

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/generate",
    json={
        "data": df.to_dict('records'),
        "n_samples": 1000,
        "method": "ctgan"
    }
)
synthetic = pd.DataFrame(response.json()["data"])
```
