# Text Generation

Generate synthetic text using LLMs with privacy protection.

## OpenAI Backend

Use GPT models via OpenAI API.

```python
import os
os.environ['OPENAI_API_KEY'] = 'your-key'

from genesis.generators.text import LLMTextGenerator

generator = LLMTextGenerator(
    backend='openai',
    model='gpt-3.5-turbo',
    temperature=0.7,
)

generator.fit(text_data)
synthetic_texts = generator.generate(n_samples=100)
```

### Parameters
- `model`: 'gpt-3.5-turbo', 'gpt-4', etc.
- `temperature`: Creativity (0.0-1.0)
- `max_tokens`: Max output length

## HuggingFace Backend

Local inference with open models.

```python
generator = LLMTextGenerator(
    backend='huggingface',
    model='gpt2',  # or 'facebook/opt-350m', etc.
    temperature=0.8,
)

generator.fit(text_data)
synthetic_texts = generator.generate(n_samples=100)
```

### Supported Models
- GPT-2
- OPT
- BLOOM
- LLaMA (requires access)

## Privacy-Safe Generation

Filter PII from generated text:

```python
generator = LLMTextGenerator(
    backend='openai',
    privacy_filter=True,
    pii_patterns=['email', 'phone', 'ssn', 'name']
)
```

## Conditional Generation

Generate text with specific attributes:

```python
# Generate positive reviews
positive = generator.generate(
    n_samples=10,
    context="positive 5-star customer review"
)

# Generate specific topics
tech_reviews = generator.generate(
    n_samples=10,
    context="electronics product review"
)
```

## Batch Generation

For large volumes:

```python
synthetic = generator.generate(
    n_samples=1000,
    batch_size=50  # Process 50 at a time
)
```
