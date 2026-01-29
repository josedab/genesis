"""Prompt templates for the synthetic data agent."""

from typing import Any, Dict, Optional


class PromptTemplates:
    """Templates for LLM prompts used by the agent."""

    SYSTEM_PROMPT = """You are a synthetic data generation assistant. Your role is to help users generate realistic synthetic datasets.

When a user describes their data needs, you must:
1. Understand the data schema they need (columns, types, relationships)
2. Identify any conditions or constraints (ranges, categories, distributions)
3. Determine the number of samples needed
4. Configure the appropriate generator settings

You respond in JSON format with the following structure:
{
    "schema": {
        "columns": [
            {"name": "column_name", "type": "numeric|categorical|datetime|text", "description": "..."}
        ]
    },
    "conditions": {
        "column_name": value,  // for equality
        "column_name": [">=", value],  // for operators
        "column_name": ["between", [min, max]]  // for ranges
    },
    "constraints": [
        {"type": "positive|range|unique|regex", "column": "name", "params": {...}}
    ],
    "n_samples": 1000,
    "generator_config": {
        "method": "auto|ctgan|tvae|gaussian_copula",
        "epochs": 300
    },
    "clarification_needed": null,  // or a string question if more info is needed
    "explanation": "Brief explanation of the configuration"
}

Available constraint types:
- positive: Ensure values are > 0
- non_negative: Ensure values are >= 0
- range: min/max bounds, params: {"min": x, "max": y}
- unique: All values must be unique
- regex: String pattern, params: {"pattern": "regex"}
- one_of: Value must be in list, params: {"values": [...]}

Available operators for conditions:
- "=", "==": Equal to
- "!=", "<>": Not equal
- ">", ">=", "<", "<=": Comparisons
- "in": Value in list
- "not_in": Value not in list
- "between": Between two values (inclusive)

Always prefer simple configurations when possible. Only add complexity if the user explicitly requests it."""

    PARSE_REQUEST_PROMPT = """Parse the following user request for synthetic data generation.

User request: {request}

{context}

Respond with a JSON configuration. If the request is unclear, set "clarification_needed" to a specific question."""

    REFINE_PROMPT = """The user provided additional information in response to your clarification.

Original request: {original_request}
Your question: {question}
User's answer: {answer}

Previous configuration:
{previous_config}

Update the configuration based on the user's answer. Respond with the complete updated JSON configuration."""

    SCHEMA_CONTEXT = """Current schema information:
{schema}

Use this schema to understand available columns and their types."""

    SAMPLE_DATA_CONTEXT = """Sample of the data (first 5 rows):
{sample}

Use this to understand the data structure and typical values."""

    @classmethod
    def get_parse_prompt(
        cls,
        request: str,
        schema: Optional[Dict[str, Any]] = None,
        sample_data: Optional[str] = None,
    ) -> str:
        """Build the prompt for parsing a user request.

        Args:
            request: User's natural language request
            schema: Optional schema information
            sample_data: Optional sample data as string

        Returns:
            Formatted prompt string
        """
        context_parts = []

        if schema:
            context_parts.append(cls.SCHEMA_CONTEXT.format(schema=schema))

        if sample_data:
            context_parts.append(cls.SAMPLE_DATA_CONTEXT.format(sample=sample_data))

        context = (
            "\n\n".join(context_parts) if context_parts else "No additional context available."
        )

        return cls.PARSE_REQUEST_PROMPT.format(
            request=request,
            context=context,
        )

    @classmethod
    def get_refine_prompt(
        cls,
        original_request: str,
        question: str,
        answer: str,
        previous_config: Dict[str, Any],
    ) -> str:
        """Build the prompt for refining configuration.

        Args:
            original_request: Original user request
            question: Clarification question that was asked
            answer: User's answer
            previous_config: Previous configuration dict

        Returns:
            Formatted prompt string
        """
        import json

        return cls.REFINE_PROMPT.format(
            original_request=original_request,
            question=question,
            answer=answer,
            previous_config=json.dumps(previous_config, indent=2),
        )
