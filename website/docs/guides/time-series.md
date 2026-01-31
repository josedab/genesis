---
sidebar_position: 2
title: Time Series
---

# Time Series Generation

Generate synthetic temporal data that preserves trends, seasonality, and autocorrelation.

## Quick Start

```python
from genesis import TimeSeriesGenerator
import pandas as pd

# Load time series data
df = pd.read_csv('stock_prices.csv', parse_dates=['date'])
df = df.set_index('date').sort_index()

# Generate synthetic time series
generator = TimeSeriesGenerator()
generator.fit(df['close'], sequence_length=20)
synthetic = generator.generate(n_sequences=10)
```

## Time Series Configuration

### Sequence Length

Controls how much history the model considers:

```python
generator.fit(
    data,
    sequence_length=50  # Look at 50 timesteps for patterns
)
```

Guidelines:
- **Short-term patterns**: 10-30 timesteps
- **Weekly seasonality**: 7 (daily) or 168 (hourly)
- **Monthly patterns**: 30 (daily) or 720 (hourly)

### Generating Sequences

```python
# Generate multiple sequences
sequences = generator.generate(n_sequences=10)

# Each sequence has shape (sequence_length, n_features)
for i, seq in enumerate(sequences):
    print(f"Sequence {i}: shape={seq.shape}")
```

## Multivariate Time Series

Generate correlated time series:

```python
# Multivariate data
df = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

generator.fit(df, sequence_length=30)
synthetic = generator.generate(n_sequences=5)

# synthetic has all columns preserved
print(synthetic[0].columns)  # ['open', 'high', 'low', 'close', 'volume']
```

## Preserving Statistical Properties

### Trend Preservation

```python
import matplotlib.pyplot as plt

# Fit on trending data
generator.fit(df, sequence_length=50)
synthetic = generator.generate(n_sequences=1)[0]

# Compare trends
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df.values[:100], label='Real')
ax.plot(synthetic.values[:100], label='Synthetic')
ax.legend()
plt.show()
```

### Seasonality

```python
# Generator learns seasonal patterns automatically
# Use sequence_length >= period for best results

# Weekly seasonality in daily data
generator.fit(data, sequence_length=14)  # 2 weeks

# Daily seasonality in hourly data
generator.fit(data, sequence_length=48)  # 2 days
```

### Autocorrelation

```python
from pandas.plotting import autocorrelation_plot

fig, axes = plt.subplots(2, 1, figsize=(12, 6))
autocorrelation_plot(df, ax=axes[0])
axes[0].set_title('Real Data ACF')
autocorrelation_plot(synthetic, ax=axes[1])
axes[1].set_title('Synthetic Data ACF')
plt.tight_layout()
```

## Conditional Time Series

Generate sequences with specific characteristics:

```python
from genesis import ConditionalTimeSeriesGenerator

generator = ConditionalTimeSeriesGenerator()
generator.fit(df, sequence_length=30)

# Generate upward trending sequences
upward = generator.generate(
    n_sequences=10,
    conditions={'trend': 'up'}
)

# Generate volatile sequences
volatile = generator.generate(
    n_sequences=10,
    conditions={'volatility': 'high'}
)
```

## Continuous Generation

Generate long continuous sequences:

```python
# Generate a single long sequence
long_sequence = generator.generate_continuous(
    n_timesteps=1000,
    seed_sequence=df.iloc[-30:].values  # Start from real data
)
```

## Financial Time Series

Specific patterns for financial data:

```python
from genesis import FinancialTimeSeriesGenerator

generator = FinancialTimeSeriesGenerator()
generator.fit(
    ohlcv_data,
    sequence_length=50,
    preserve_constraints=True  # high >= open, low <= open, etc.
)

synthetic = generator.generate(n_sequences=100)

# Constraints are preserved
assert (synthetic['high'] >= synthetic['low']).all()
```

## Sensor Data

For IoT/sensor time series:

```python
# Multiple sensors
sensor_df = pd.DataFrame({
    'temp_sensor_1': [...],
    'temp_sensor_2': [...],
    'pressure': [...],
    'timestamp': [...]
})

generator.fit(sensor_df[['temp_sensor_1', 'temp_sensor_2', 'pressure']])
synthetic = generator.generate(n_sequences=50)
```

## Evaluation

```python
from genesis.evaluation import TimeSeriesMetrics

metrics = TimeSeriesMetrics(real_sequences, synthetic_sequences)

print(f"DTW Distance: {metrics.dtw_distance():.3f}")
print(f"ACF Similarity: {metrics.acf_similarity():.3f}")
print(f"Trend Similarity: {metrics.trend_similarity():.3f}")
print(f"Distribution Score: {metrics.marginal_similarity():.3f}")
```

## Complete Example

```python
import pandas as pd
import numpy as np
from genesis import TimeSeriesGenerator

# Load energy consumption data
df = pd.read_csv('energy_consumption.csv', parse_dates=['timestamp'])
df = df.set_index('timestamp').sort_index()

# Select features
features = ['consumption_kwh', 'temperature', 'humidity']
data = df[features]

# Create generator
generator = TimeSeriesGenerator(
    config={
        'hidden_dim': 64,
        'n_layers': 2,
        'epochs': 200
    }
)

# Fit with 24-hour window (hourly data)
generator.fit(data, sequence_length=24)

# Generate 100 daily patterns
synthetic = generator.generate(n_sequences=100)

# Evaluate
from genesis.evaluation import TimeSeriesMetrics
metrics = TimeSeriesMetrics(data.values[:100], np.array([s.values for s in synthetic]))
print(f"Quality Score: {metrics.overall_score():.1%}")

# Save
for i, seq in enumerate(synthetic):
    seq.to_csv(f'synthetic_energy_{i}.csv')
```

## Best Practices

1. **Choose sequence_length carefully** - Should capture at least one full cycle
2. **Normalize data** - Time series generators work best with scaled data
3. **Check stationarity** - Consider differencing non-stationary data
4. **Validate ACF/PACF** - Key indicator of temporal structure preservation
5. **Use domain constraints** - E.g., non-negative values for counts

## Troubleshooting

### Generated values are constant
- Increase `sequence_length`
- Increase training epochs
- Check if original data has variance

### Poor autocorrelation preservation
- Use longer `sequence_length`
- Try different model architecture
- Ensure data isn't too noisy

### Mode collapse (all sequences identical)
- Increase model diversity settings
- Add noise to training
- Try different random seeds

## Next Steps

- **[Conditional Generation](/docs/guides/conditional-generation)** - Scenario-based generation
- **[Drift Detection](/docs/guides/drift-detection)** - Monitor time series changes
- **[Streaming](/docs/guides/streaming)** - Real-time generation
