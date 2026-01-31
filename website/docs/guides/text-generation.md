---
sidebar_position: 3
title: Text Generation
---

# Text Generation

Generate realistic synthetic text that preserves linguistic patterns and domain vocabulary.

## Quick Start

```python
from genesis import TextGenerator

# Generate product descriptions
generator = TextGenerator()
generator.fit(product_descriptions)  # List of strings
synthetic = generator.generate(100)
```

## Configuration

```python
generator = TextGenerator(
    config={
        'model': 'lstm',           # 'lstm', 'gru', 'markov', 'gpt2'
        'max_length': 200,         # Maximum generated length
        'temperature': 0.8,        # Creativity (0.1-1.5)
        'min_length': 20           # Minimum generated length
    }
)
```

### Model Options

| Model | Best For | Speed | Quality |
|-------|----------|-------|---------|
| `markov` | Short text, fast generation | ⚡⚡⚡ | ⭐⭐ |
| `lstm` | General purpose | ⚡⚡ | ⭐⭐⭐ |
| `gru` | Similar to LSTM, faster | ⚡⚡ | ⭐⭐⭐ |
| `gpt2` | High quality, longer text | ⚡ | ⭐⭐⭐⭐ |

## Temperature

Controls randomness/creativity:

```python
# Low temperature (0.3) - More predictable
generator.generate(10, temperature=0.3)

# High temperature (1.2) - More creative/varied
generator.generate(10, temperature=1.2)
```

- **0.1-0.4**: Conservative, repetitive
- **0.5-0.8**: Balanced (recommended)
- **0.9-1.5**: Creative, may have errors

## Domain-Specific Text

### Product Descriptions

```python
product_texts = [
    "Premium leather wallet with RFID blocking technology",
    "Wireless bluetooth headphones with noise cancellation",
    # ... more examples
]

generator = TextGenerator(config={'model': 'lstm'})
generator.fit(product_texts)

# Generate new product descriptions
synthetic = generator.generate(50)
```

### Customer Reviews

```python
reviews = [
    "Great product! Fast shipping and exactly as described.",
    "Good quality but took longer to arrive than expected.",
    # ... more examples
]

generator.fit(reviews)
synthetic_reviews = generator.generate(100)
```

### Technical Documentation

```python
from genesis import TextGenerator

docs = open('api_docs.txt').read().split('\n\n')

generator = TextGenerator(
    config={
        'model': 'gpt2',
        'max_length': 500
    }
)
generator.fit(docs)
synthetic_docs = generator.generate(20)
```

## Conditional Text Generation

Generate text with specific attributes:

```python
from genesis import ConditionalTextGenerator

# Reviews with sentiment
data = [
    {'text': 'Amazing product!', 'sentiment': 'positive'},
    {'text': 'Disappointing quality.', 'sentiment': 'negative'},
    # ...
]

generator = ConditionalTextGenerator()
generator.fit(data)

# Generate positive reviews
positive = generator.generate(50, conditions={'sentiment': 'positive'})

# Generate negative reviews
negative = generator.generate(50, conditions={'sentiment': 'negative'})
```

## Structured Text

Generate text with specific structure:

```python
# Email templates
emails = [
    "Subject: Order Confirmation\n\nDear Customer,\n\nThank you for...",
    "Subject: Shipping Update\n\nHello,\n\nYour order has shipped...",
]

generator = TextGenerator(config={'preserve_structure': True})
generator.fit(emails)
synthetic_emails = generator.generate(100)
```

## Text with Entities

Generate text with placeholder entities:

```python
from genesis import EntityAwareTextGenerator

texts = [
    "Customer John Smith placed order #12345 on January 1st.",
    "Customer Jane Doe placed order #67890 on February 15th.",
]

generator = EntityAwareTextGenerator()
generator.fit(texts)

# Entities are replaced with realistic synthetic values
synthetic = generator.generate(100)
# "Customer Michael Brown placed order #45678 on March 22nd."
```

## Multi-Language

```python
# Genesis supports multiple languages
french_texts = [
    "Excellent produit, je recommande vivement!",
    "Livraison rapide et conforme à la description.",
]

generator = TextGenerator()
generator.fit(french_texts)
synthetic_french = generator.generate(50)
```

## Evaluation

```python
from genesis.evaluation import TextMetrics

metrics = TextMetrics(real_texts, synthetic_texts)

print(f"Perplexity: {metrics.perplexity():.2f}")
print(f"BLEU Score: {metrics.bleu_score():.3f}")
print(f"Vocabulary Overlap: {metrics.vocabulary_overlap():.1%}")
print(f"Avg Length Match: {metrics.length_similarity():.1%}")
```

## Privacy Considerations

Text generation can memorize sensitive data:

```python
from genesis import TextGenerator

generator = TextGenerator(
    config={
        'privacy': {
            'epsilon': 1.0,           # Differential privacy
            'remove_pii': True,       # Remove PII patterns
            'min_frequency': 3        # Words must appear 3+ times
        }
    }
)
```

## Complete Example

```python
import pandas as pd
from genesis import TextGenerator
from genesis.evaluation import TextMetrics

# Load customer support tickets
df = pd.read_csv('support_tickets.csv')
tickets = df['description'].tolist()

# Create generator
generator = TextGenerator(
    config={
        'model': 'lstm',
        'max_length': 300,
        'temperature': 0.7
    }
)

# Train
generator.fit(tickets)

# Generate 1000 synthetic tickets
synthetic = generator.generate(1000)

# Evaluate
metrics = TextMetrics(tickets[:1000], synthetic)
print(f"Quality Score: {metrics.overall_score():.1%}")

# Save
pd.DataFrame({'description': synthetic}).to_csv('synthetic_tickets.csv')
```

## Best Practices

1. **Clean input text** - Remove noise, fix encoding issues
2. **Use sufficient training data** - 1000+ examples recommended
3. **Match temperature to use case** - Lower for formal text
4. **Check for memorization** - Verify no exact copies
5. **Validate domain vocabulary** - Important terms should appear

## Troubleshooting

### Generated text is gibberish
- Lower the temperature
- Increase training data
- Use a more powerful model (gpt2)

### Text is too repetitive
- Increase temperature
- Add more diverse training examples
- Check for dominant patterns in training data

### Missing domain terms
- Ensure terms appear multiple times in training data
- Lower the `min_frequency` threshold
- Use conditional generation to force topics

## Next Steps

- **[Domain Generators](/docs/guides/domain-generators)** - Pre-built generators for specific domains
- **[Conditional Generation](/docs/guides/conditional-generation)** - Control generation attributes
- **[Privacy](/docs/concepts/privacy)** - Protect sensitive text data
