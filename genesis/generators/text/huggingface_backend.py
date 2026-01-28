"""HuggingFace backend for text generation."""

from typing import List, Optional

from genesis.core.config import TextGenerationConfig
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class HuggingFaceBackend:
    """HuggingFace Transformers backend for text generation."""

    def __init__(
        self,
        config: Optional[TextGenerationConfig] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize HuggingFace backend.

        Args:
            config: Text generation configuration
            model_name: Name of model to use (overrides config)
        """
        self.config = config or TextGenerationConfig()
        self.model_name = model_name or self.config.model_name

        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Load model and tokenizer."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers package required. Install with: pip install genesis-synth[llm]"
            ) from e

        logger.info(f"Loading model: {self.model_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Set pad token if not exists
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def generate(
        self,
        prompt: str,
        n_samples: int = 1,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """Generate text using HuggingFace model.

        Args:
            prompt: Prompt for generation
            n_samples: Number of samples to generate
            max_tokens: Maximum new tokens per sample
            temperature: Sampling temperature

        Returns:
            List of generated texts
        """
        self._load_model()

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        # Tokenize prompt
        inputs = self._tokenizer(prompt, return_tensors="pt", padding=True)

        # Generate
        outputs = self._model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            num_return_sequences=n_samples,
            temperature=temperature,
            do_sample=True,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            pad_token_id=self._tokenizer.pad_token_id,
        )

        # Decode
        results = []
        for output in outputs:
            text = self._tokenizer.decode(output, skip_special_tokens=True)
            # Remove the prompt from the output
            if text.startswith(prompt):
                text = text[len(prompt) :].strip()
            results.append(text)

        return results

    def generate_similar(
        self,
        examples: List[str],
        n_samples: int = 1,
        context: Optional[str] = None,
    ) -> List[str]:
        """Generate text similar to provided examples.

        Args:
            examples: Example texts to base generation on
            n_samples: Number of samples to generate
            context: Additional context for generation

        Returns:
            List of generated texts
        """
        # Use examples as prompts with continuation
        results = []

        for _i in range(n_samples):
            # Pick a random example as seed
            import random

            example = random.choice(examples)

            # Use first part as prompt
            prompt_len = min(len(example) // 2, 50)
            prompt = example[:prompt_len]

            generated = self.generate(prompt, n_samples=1, max_tokens=self.config.max_tokens)
            if generated:
                results.append(generated[0])

        return results

    def fine_tune(
        self,
        texts: List[str],
        epochs: int = 3,
        batch_size: int = 4,
    ) -> None:
        """Fine-tune the model on provided texts.

        Args:
            texts: Training texts
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        self._load_model()

        try:
            from datasets import Dataset
            from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
        except ImportError as e:
            raise ImportError("transformers and datasets packages required") from e

        logger.info(f"Fine-tuning on {len(texts)} texts for {epochs} epochs")

        # Create dataset
        dataset = Dataset.from_dict({"text": texts})

        def tokenize_function(examples):
            return self._tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length",
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer,
            mlm=False,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./fine_tuned_model",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            logging_steps=100,
            save_steps=500,
            warmup_steps=100,
        )

        # Trainer
        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        trainer.train()
        logger.info("Fine-tuning completed")
