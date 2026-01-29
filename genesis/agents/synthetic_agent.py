"""Synthetic Data Agent - Natural language interface for data generation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from genesis.agents.parser import ConfigParser, GenerationConfig
from genesis.agents.prompts import PromptTemplates
from genesis.core.base import SyntheticGenerator
from genesis.core.config import PrivacyConfig
from genesis.core.exceptions import ValidationError
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    role: str  # "user" or "assistant"
    content: str
    config: Optional[GenerationConfig] = None


@dataclass
class AgentState:
    """State of the agent conversation."""

    history: List[ConversationTurn] = field(default_factory=list)
    current_config: Optional[GenerationConfig] = None
    original_request: Optional[str] = None
    is_complete: bool = False
    generated_data: Optional[pd.DataFrame] = None


class SyntheticDataAgent:
    """LLM-powered agent for natural language synthetic data generation.

    This agent allows users to describe their data needs in natural language
    and automatically configures and runs the appropriate generators.

    Example:
        >>> agent = SyntheticDataAgent(api_key="sk-...")
        >>> response = agent.request(
        ...     "Generate 10,000 customer records with realistic names, "
        ...     "ages between 18-65, US addresses, and income normally "
        ...     "distributed around $50k"
        ... )
        >>> if response.needs_clarification:
        ...     response = agent.respond(user_input)
        >>> data = response.data
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        base_data: Optional[pd.DataFrame] = None,
        privacy: Optional[PrivacyConfig] = None,
        temperature: float = 0.2,
        max_retries: int = 3,
    ) -> None:
        """Initialize the agent.

        Args:
            api_key: API key for LLM provider
            model: Model to use (e.g., "gpt-4o-mini", "gpt-4o", "claude-3-sonnet")
            provider: LLM provider ("openai", "anthropic")
            base_data: Optional DataFrame to use as reference for schema
            privacy: Privacy configuration for generated data
            temperature: LLM temperature (lower = more deterministic)
            max_retries: Maximum parse retries on errors
        """
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.base_data = base_data
        self.privacy = privacy or PrivacyConfig()
        self.temperature = temperature
        self.max_retries = max_retries

        self._parser = ConfigParser()
        self._state = AgentState()
        self._llm_client: Optional[Any] = None

        # Schema info derived from base_data
        self._schema_info: Optional[Dict[str, Any]] = None
        self._sample_str: Optional[str] = None

        if base_data is not None:
            self._analyze_base_data(base_data)

    def _analyze_base_data(self, data: pd.DataFrame) -> None:
        """Analyze base data to extract schema information."""
        self._schema_info = {
            "columns": [
                {
                    "name": col,
                    "type": self._infer_type(data[col]),
                    "n_unique": int(data[col].nunique()),
                    "sample_values": data[col].dropna().head(3).tolist(),
                }
                for col in data.columns
            ],
            "n_rows": len(data),
            "n_columns": len(data.columns),
        }
        self._sample_str = data.head(5).to_string()

    def _infer_type(self, series: pd.Series) -> str:
        """Infer column type from pandas series."""
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        elif pd.api.types.is_bool_dtype(series):
            return "boolean"
        else:
            n_unique = series.nunique()
            if n_unique < 20 or n_unique < len(series) * 0.05:
                return "categorical"
            return "text"

    def _get_llm_client(self) -> Any:
        """Get or create LLM client."""
        if self._llm_client is not None:
            return self._llm_client

        if self.provider == "openai":
            try:
                import openai

                self._llm_client = openai.OpenAI(api_key=self.api_key)
            except ImportError as e:
                raise ImportError(
                    "OpenAI package not installed. " "Install with: pip install genesis[llm]"
                ) from e
        elif self.provider == "anthropic":
            try:
                import anthropic

                self._llm_client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError as e:
                raise ImportError(
                    "Anthropic package not installed. " "Install with: pip install anthropic"
                ) from e
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        return self._llm_client

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM and return response.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt

        Returns:
            LLM response text
        """
        client = self._get_llm_client()

        if self.provider == "openai":
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
            )
            return response.content[0].text

        raise ValueError(f"Unknown provider: {self.provider}")

    def request(
        self,
        prompt: str,
        n_samples: Optional[int] = None,
    ) -> "AgentResponse":
        """Process a natural language request for synthetic data.

        Args:
            prompt: Natural language description of desired data
            n_samples: Override number of samples (optional)

        Returns:
            AgentResponse with config, clarification question, or data
        """
        self._state = AgentState(original_request=prompt)

        # Build prompt
        user_prompt = PromptTemplates.get_parse_prompt(
            request=prompt,
            schema=self._schema_info,
            sample_data=self._sample_str,
        )

        # Call LLM
        response_text = self._call_llm(
            PromptTemplates.SYSTEM_PROMPT,
            user_prompt,
        )

        # Parse response
        try:
            config = self._parser.parse(response_text)
        except ValidationError as e:
            logger.warning(f"Parse error: {e}, retrying...")
            # Retry with simpler prompt
            response_text = self._call_llm(
                PromptTemplates.SYSTEM_PROMPT,
                f"{user_prompt}\n\nIMPORTANT: Respond with valid JSON only.",
            )
            config = self._parser.parse(response_text)

        # Override n_samples if provided
        if n_samples is not None:
            config.n_samples = n_samples

        self._state.current_config = config
        self._state.history.append(ConversationTurn("user", prompt))
        self._state.history.append(
            ConversationTurn(
                "assistant",
                config.explanation or "Configuration ready.",
                config=config,
            )
        )

        return self._build_response()

    def respond(self, answer: str) -> "AgentResponse":
        """Respond to a clarification question.

        Args:
            answer: User's answer to the clarification question

        Returns:
            Updated AgentResponse
        """
        if self._state.current_config is None:
            raise ValidationError("No active conversation. Call request() first.")

        if self._state.current_config.is_complete:
            raise ValidationError("Configuration already complete. No clarification needed.")

        # Build refine prompt
        user_prompt = PromptTemplates.get_refine_prompt(
            original_request=self._state.original_request or "",
            question=self._state.current_config.clarification_needed or "",
            answer=answer,
            previous_config=self._state.current_config.raw_config,
        )

        # Call LLM
        response_text = self._call_llm(
            PromptTemplates.SYSTEM_PROMPT,
            user_prompt,
        )

        # Parse updated config
        config = self._parser.parse(response_text)

        self._state.current_config = config
        self._state.history.append(ConversationTurn("user", answer))
        self._state.history.append(
            ConversationTurn(
                "assistant",
                config.explanation or "Configuration updated.",
                config=config,
            )
        )

        return self._build_response()

    def generate(self) -> pd.DataFrame:
        """Generate synthetic data using current configuration.

        Returns:
            Generated DataFrame

        Raises:
            ValidationError: If configuration is not complete
        """
        if self._state.current_config is None:
            raise ValidationError("No configuration available. Call request() first.")

        if not self._state.current_config.is_complete:
            raise ValidationError(
                f"Configuration incomplete. Please answer: "
                f"{self._state.current_config.clarification_needed}"
            )

        config = self._state.current_config

        # Create generator
        generator = SyntheticGenerator(
            method=config.generator_method,
            config=config.to_generator_config(),
            privacy=self.privacy,
        )

        # Use base_data if available, otherwise need sample data
        if self.base_data is not None:
            generator.fit(
                self.base_data,
                constraints=config.to_constraints(),
            )
        else:
            raise ValidationError(
                "No base data provided. Either provide base_data in __init__ "
                "or use generate_from_schema() for schema-only generation."
            )

        # Generate with conditions if specified
        if config.conditions:
            from genesis.generators.conditional import ConditionalSampler

            sampler = ConditionalSampler()
            data = sampler.sample(
                generator_fn=lambda n: generator.generate(n),
                n_samples=config.n_samples,
                conditions=config.to_conditions_dict(),
            )
        else:
            data = generator.generate(config.n_samples)

        self._state.generated_data = data
        self._state.is_complete = True

        return data

    def _build_response(self) -> "AgentResponse":
        """Build response from current state."""
        return AgentResponse(
            config=self._state.current_config,
            needs_clarification=not self._state.current_config.is_complete,
            clarification_question=self._state.current_config.clarification_needed,
            explanation=self._state.current_config.explanation,
            data=self._state.generated_data,
            agent=self,
        )

    def reset(self) -> None:
        """Reset the agent state."""
        self._state = AgentState()

    @property
    def history(self) -> List[ConversationTurn]:
        """Get conversation history."""
        return self._state.history

    @property
    def current_config(self) -> Optional[GenerationConfig]:
        """Get current configuration."""
        return self._state.current_config


@dataclass
class AgentResponse:
    """Response from the synthetic data agent."""

    config: Optional[GenerationConfig]
    needs_clarification: bool
    clarification_question: Optional[str]
    explanation: str
    data: Optional[pd.DataFrame]
    agent: SyntheticDataAgent

    def respond(self, answer: str) -> "AgentResponse":
        """Respond to clarification question."""
        return self.agent.respond(answer)

    def generate(self) -> pd.DataFrame:
        """Generate data with current configuration."""
        return self.agent.generate()

    def __repr__(self) -> str:
        if self.needs_clarification:
            return (
                f"AgentResponse(needs_clarification=True, question='{self.clarification_question}')"
            )
        elif self.data is not None:
            return f"AgentResponse(data={self.data.shape})"
        else:
            return f"AgentResponse(config_ready=True, n_samples={self.config.n_samples})"


def chat(
    prompt: str,
    base_data: Optional[pd.DataFrame] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    n_samples: Optional[int] = None,
) -> AgentResponse:
    """Quick interface for one-shot synthetic data generation.

    Args:
        prompt: Natural language description of desired data
        base_data: Optional DataFrame to use as reference
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        model: Model to use
        n_samples: Number of samples to generate

    Returns:
        AgentResponse with configuration ready

    Example:
        >>> response = chat(
        ...     "Generate customer data with balanced fraud labels",
        ...     base_data=real_customers,
        ... )
        >>> data = response.generate()
    """
    import os

    api_key = api_key or os.environ.get("OPENAI_API_KEY")

    agent = SyntheticDataAgent(
        api_key=api_key,
        model=model,
        base_data=base_data,
    )

    return agent.request(prompt, n_samples=n_samples)
