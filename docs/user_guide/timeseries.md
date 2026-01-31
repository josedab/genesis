# Time Series Generation

Generate synthetic time series data while preserving temporal patterns.

## Statistical Generator

Fast statistical methods for time series.

```python
from genesis.generators.timeseries import StatisticalTimeSeriesGenerator

generator = StatisticalTimeSeriesGenerator()
generator.fit(data)
synthetic = generator.generate(n_samples=500)
```

### Features
- ARIMA modeling
- Seasonal decomposition
- Trend preservation
- Noise modeling

## TimeGAN

Deep learning for complex temporal patterns.

```python
from genesis.generators.timeseries import TimeGANGenerator

generator = TimeGANGenerator(
    seq_len=24,      # Sequence length
    n_epochs=100,    # Training epochs
    hidden_dim=24,   # Hidden dimension
)
generator.fit(data)
synthetic = generator.generate(n_samples=500)
```

### Architecture
- Embedding network
- Recovery network
- Generator network
- Discriminator network

## Multivariate Time Series

Handle multiple correlated time series:

```python
data = pd.DataFrame({
    'temperature': [...],
    'humidity': [...],
    'pressure': [...]
})

generator.fit(data)  # Learns cross-series correlations
synthetic = generator.generate(500)
```

## Time Features

Add time-based features:

```python
# With explicit timestamp
data['timestamp'] = pd.date_range('2023-01-01', periods=len(data), freq='H')
generator.fit(data, time_column='timestamp')
```
