"""LLM-Powered Schema Inference.

This module uses Large Language Models to intelligently infer data constraints,
distributions, and relationships from column names and sample data.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class InferredConstraint:
    """A constraint inferred by LLM analysis."""

    column: str
    constraint_type: str  # "range", "pattern", "values", "distribution", "nullable"
    parameters: Dict[str, Any]
    confidence: float  # 0-1
    reasoning: str


@dataclass
class InferredRelationship:
    """A relationship inferred between columns."""

    source_column: str
    target_column: str
    relationship_type: str  # "derived", "correlated", "dependent", "foreign_key"
    description: str
    confidence: float


@dataclass
class InferredSchema:
    """Complete schema inferred by LLM."""

    columns: Dict[str, Dict[str, Any]]
    constraints: List[InferredConstraint]
    relationships: List[InferredRelationship]
    data_domain: str
    suggested_generation_method: str
    notes: List[str] = field(default_factory=list)

    def to_genesis_config(self) -> Dict[str, Any]:
        """Convert to Genesis configuration format."""
        config = {
            "columns": {},
            "constraints": [],
        }

        for col_name, col_info in self.columns.items():
            config["columns"][col_name] = {
                "type": col_info.get("inferred_type", "unknown"),
                "nullable": col_info.get("nullable", True),
                "description": col_info.get("description", ""),
            }

            if "distribution" in col_info:
                config["columns"][col_name]["distribution"] = col_info["distribution"]

            if "valid_values" in col_info:
                config["columns"][col_name]["valid_values"] = col_info["valid_values"]

        for constraint in self.constraints:
            config["constraints"].append(
                {
                    "column": constraint.column,
                    "type": constraint.constraint_type,
                    "parameters": constraint.parameters,
                }
            )

        return config


class LLMSchemaInferrer:
    """Infer schema using LLM analysis.

    Can use OpenAI GPT-4, Anthropic Claude, or fall back to
    rule-based heuristics if no LLM is available.
    """

    # Common column name patterns and their likely semantics
    COLUMN_PATTERNS = {
        # Identifiers
        r"^id$|_id$|^pk$": {"type": "id", "description": "Unique identifier"},
        r"^uuid$|_uuid$": {"type": "uuid", "description": "UUID identifier"},
        # Personal info
        r"^name$|_name$|^first_?name$|^last_?name$": {
            "type": "name",
            "description": "Person's name",
        },
        r"^email$|_email$|^e_?mail$": {"type": "email", "description": "Email address"},
        r"^phone$|_phone$|^tel$|^mobile$": {"type": "phone", "description": "Phone number"},
        r"^age$|_age$": {
            "type": "integer",
            "description": "Age in years",
            "constraints": {"min": 0, "max": 120},
        },
        r"^gender$|^sex$": {
            "type": "category",
            "description": "Gender",
            "values": ["Male", "Female", "Other", "Unknown"],
        },
        # Location
        r"^address$|_address$|^street$": {"type": "address", "description": "Street address"},
        r"^city$|_city$": {"type": "city", "description": "City name"},
        r"^state$|_state$|^province$": {"type": "state", "description": "State/Province"},
        r"^country$|_country$": {"type": "country", "description": "Country"},
        r"^zip$|^postal|_zip$|_postal$": {"type": "postal_code", "description": "Postal/ZIP code"},
        r"^lat$|^latitude$": {
            "type": "float",
            "description": "Latitude",
            "constraints": {"min": -90, "max": 90},
        },
        r"^lon$|^lng$|^longitude$": {
            "type": "float",
            "description": "Longitude",
            "constraints": {"min": -180, "max": 180},
        },
        # Dates and times
        r"^date$|_date$|^dt$": {"type": "date", "description": "Date"},
        r"^time$|_time$|^timestamp$|_at$": {"type": "datetime", "description": "Date and time"},
        r"^created|^updated|^modified": {"type": "datetime", "description": "Record timestamp"},
        r"^year$|_year$": {
            "type": "integer",
            "description": "Year",
            "constraints": {"min": 1900, "max": 2100},
        },
        r"^month$|_month$": {
            "type": "integer",
            "description": "Month",
            "constraints": {"min": 1, "max": 12},
        },
        r"^day$|_day$": {
            "type": "integer",
            "description": "Day of month",
            "constraints": {"min": 1, "max": 31},
        },
        # Financial
        r"^price$|_price$|^cost$|_cost$|^amount$|_amount$": {
            "type": "float",
            "description": "Monetary amount",
            "constraints": {"min": 0},
        },
        r"^salary$|^income$|^wage$": {
            "type": "float",
            "description": "Income/salary",
            "constraints": {"min": 0},
        },
        r"^currency$|_currency$": {
            "type": "category",
            "description": "Currency code",
            "values": ["USD", "EUR", "GBP", "JPY", "CNY"],
        },
        # Counts and quantities
        r"^count$|_count$|^num_|^n_|^total$|_total$": {
            "type": "integer",
            "description": "Count/quantity",
            "constraints": {"min": 0},
        },
        r"^quantity$|^qty$|_qty$": {
            "type": "integer",
            "description": "Quantity",
            "constraints": {"min": 0},
        },
        # Percentages and ratios
        r"^pct$|^percent|_pct$|_percent|^ratio$|_ratio$|^rate$|_rate$": {
            "type": "float",
            "description": "Percentage/ratio",
            "constraints": {"min": 0, "max": 100},
        },
        # Status and categories
        r"^status$|_status$": {"type": "category", "description": "Status field"},
        r"^type$|_type$|^category$|_category$": {
            "type": "category",
            "description": "Category/type",
        },
        r"^is_|^has_|^can_|^should_": {"type": "boolean", "description": "Boolean flag"},
        # Text
        r"^desc$|^description$|_desc$|_description$": {
            "type": "text",
            "description": "Description text",
        },
        r"^comment$|^note$|^remarks$": {"type": "text", "description": "Comment/notes"},
    }

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize schema inferrer.

        Args:
            llm_provider: "openai", "anthropic", or None for rule-based
            api_key: API key for LLM provider
            model: Model name to use
        """
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.model = model or self._default_model(llm_provider)

        self._llm_client = None
        if llm_provider and api_key:
            self._init_llm_client()

    def _default_model(self, provider: Optional[str]) -> str:
        """Get default model for provider."""
        if provider == "openai":
            return "gpt-4"
        elif provider == "anthropic":
            return "claude-3-sonnet-20240229"
        return ""

    def _init_llm_client(self):
        """Initialize LLM client."""
        try:
            if self.llm_provider == "openai":
                import openai

                self._llm_client = openai.OpenAI(api_key=self.api_key)
            elif self.llm_provider == "anthropic":
                import anthropic

                self._llm_client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            self._llm_client = None

    def infer(
        self,
        data: pd.DataFrame,
        sample_size: int = 100,
        use_llm: bool = True,
    ) -> InferredSchema:
        """Infer schema from data.

        Args:
            data: Input DataFrame
            sample_size: Number of rows to sample for analysis
            use_llm: Whether to use LLM (if available)

        Returns:
            InferredSchema with column info and constraints
        """
        # Sample data
        if len(data) > sample_size:
            sample = data.sample(n=sample_size, random_state=42)
        else:
            sample = data

        # Try LLM inference
        if use_llm and self._llm_client:
            try:
                return self._infer_with_llm(sample)
            except Exception as e:
                print(f"LLM inference failed: {e}, falling back to rules")

        # Fall back to rule-based inference
        return self._infer_with_rules(sample)

    def _infer_with_llm(self, sample: pd.DataFrame) -> InferredSchema:
        """Infer schema using LLM."""
        # Build prompt
        prompt = self._build_llm_prompt(sample)

        if self.llm_provider == "openai":
            response = self._llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data schema expert. Analyze the data and infer constraints and relationships.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)

        elif self.llm_provider == "anthropic":
            response = self._llm_client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            # Extract JSON from response
            content = response.content[0].text
            result = json.loads(self._extract_json(content))

        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")

        return self._parse_llm_response(result, sample)

    def _build_llm_prompt(self, sample: pd.DataFrame) -> str:
        """Build prompt for LLM."""
        # Get column info
        col_info = []
        for col in sample.columns:
            info = {
                "name": col,
                "dtype": str(sample[col].dtype),
                "n_unique": int(sample[col].nunique()),
                "n_missing": int(sample[col].isnull().sum()),
                "sample_values": sample[col].dropna().head(5).tolist(),
            }

            if pd.api.types.is_numeric_dtype(sample[col]):
                info["min"] = float(sample[col].min()) if not sample[col].isna().all() else None
                info["max"] = float(sample[col].max()) if not sample[col].isna().all() else None
                info["mean"] = float(sample[col].mean()) if not sample[col].isna().all() else None

            col_info.append(info)

        prompt = f"""Analyze this dataset and infer the schema.

Column Information:
{json.dumps(col_info, indent=2, default=str)}

Please return a JSON object with:
1. "columns": For each column, infer:
   - "inferred_type": semantic type (e.g., "email", "age", "price", "category")
   - "description": what this column represents
   - "nullable": whether it can be null
   - "distribution": suggested distribution if numeric
   - "valid_values": list of valid values if categorical
   - "constraints": any constraints (min, max, pattern)

2. "relationships": List of relationships between columns:
   - "source_column", "target_column"
   - "relationship_type": "derived", "correlated", "dependent", "foreign_key"
   - "description"

3. "data_domain": What domain this data is from (e.g., "healthcare", "finance", "retail")

4. "suggested_generation_method": Best method for synthetic generation

5. "notes": Any important observations

Return valid JSON only."""

        return prompt

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text response."""
        # Try to find JSON block
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return match.group()
        return text

    def _parse_llm_response(self, result: Dict[str, Any], sample: pd.DataFrame) -> InferredSchema:
        """Parse LLM response into InferredSchema."""
        columns = result.get("columns", {})

        # Convert constraints
        constraints = []
        for col_name, col_info in columns.items():
            if "constraints" in col_info:
                constraints.append(
                    InferredConstraint(
                        column=col_name,
                        constraint_type="range" if "min" in col_info["constraints"] else "custom",
                        parameters=col_info["constraints"],
                        confidence=0.8,
                        reasoning="Inferred by LLM",
                    )
                )

        # Convert relationships
        relationships = []
        for rel in result.get("relationships", []):
            relationships.append(
                InferredRelationship(
                    source_column=rel.get("source_column", ""),
                    target_column=rel.get("target_column", ""),
                    relationship_type=rel.get("relationship_type", "unknown"),
                    description=rel.get("description", ""),
                    confidence=0.7,
                )
            )

        return InferredSchema(
            columns=columns,
            constraints=constraints,
            relationships=relationships,
            data_domain=result.get("data_domain", "unknown"),
            suggested_generation_method=result.get(
                "suggested_generation_method", "gaussian_copula"
            ),
            notes=result.get("notes", []),
        )

    def _infer_with_rules(self, sample: pd.DataFrame) -> InferredSchema:
        """Infer schema using rule-based heuristics."""
        columns = {}
        constraints = []
        relationships = []

        for col in sample.columns:
            col_info = self._infer_column_rules(col, sample[col])
            columns[col] = col_info

            # Add constraints
            if "constraints" in col_info:
                constraints.append(
                    InferredConstraint(
                        column=col,
                        constraint_type="range",
                        parameters=col_info["constraints"],
                        confidence=0.7,
                        reasoning="Inferred from column name pattern and data",
                    )
                )

        # Detect relationships
        relationships = self._detect_relationships_rules(sample, columns)

        # Determine domain
        data_domain = self._infer_domain(columns)

        # Suggest method
        suggested_method = self._suggest_method(sample, columns)

        return InferredSchema(
            columns=columns,
            constraints=constraints,
            relationships=relationships,
            data_domain=data_domain,
            suggested_generation_method=suggested_method,
        )

    def _infer_column_rules(self, col_name: str, series: pd.Series) -> Dict[str, Any]:
        """Infer column info using rules."""
        col_lower = col_name.lower()

        # Check patterns
        for pattern, info in self.COLUMN_PATTERNS.items():
            if re.search(pattern, col_lower, re.IGNORECASE):
                result = {
                    "inferred_type": info["type"],
                    "description": info["description"],
                    "nullable": series.isnull().any(),
                }

                if "constraints" in info:
                    result["constraints"] = info["constraints"].copy()

                if "values" in info:
                    result["valid_values"] = info["values"]

                return result

        # Infer from data type and content
        if pd.api.types.is_bool_dtype(series):
            return {
                "inferred_type": "boolean",
                "description": "Boolean flag",
                "nullable": series.isnull().any(),
            }

        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            constraints = {}

            if len(non_null) > 0:
                if (non_null >= 0).all():
                    constraints["min"] = 0
                if (non_null <= 1).all():
                    constraints["max"] = 1
                elif (non_null <= 100).all() and (non_null >= 0).all():
                    constraints["max"] = 100

            return {
                "inferred_type": "integer" if pd.api.types.is_integer_dtype(series) else "float",
                "description": "Numeric value",
                "nullable": series.isnull().any(),
                "constraints": constraints if constraints else None,
                "distribution": self._detect_distribution(non_null),
            }

        if pd.api.types.is_datetime64_any_dtype(series):
            return {
                "inferred_type": "datetime",
                "description": "Date/time value",
                "nullable": series.isnull().any(),
            }

        # Categorical
        n_unique = series.nunique()
        if n_unique < 20:
            return {
                "inferred_type": "category",
                "description": "Categorical value",
                "nullable": series.isnull().any(),
                "valid_values": series.dropna().unique().tolist(),
            }

        return {
            "inferred_type": "text",
            "description": "Text value",
            "nullable": series.isnull().any(),
        }

    def _detect_distribution(self, series: pd.Series) -> Optional[str]:
        """Detect likely distribution of numeric series."""
        if len(series) < 10:
            return None

        try:
            skew = series.skew()
            kurt = series.kurtosis()

            if abs(skew) < 0.5 and abs(kurt) < 1:
                return "normal"
            elif skew > 1:
                return "exponential"
            elif (series >= 0).all() and (series <= 1).all():
                return "beta"
            elif (series == series.astype(int)).all() and (series >= 0).all():
                return "poisson"
            else:
                return "empirical"
        except Exception:
            return None

    def _detect_relationships_rules(
        self,
        sample: pd.DataFrame,
        columns: Dict[str, Dict[str, Any]],
    ) -> List[InferredRelationship]:
        """Detect relationships using rules."""
        relationships = []

        # Check for potential foreign keys
        for col in sample.columns:
            col_lower = col.lower()

            # Pattern: table_id might reference table
            if col_lower.endswith("_id"):
                base = col_lower[:-3]
                for other_col in sample.columns:
                    if other_col.lower() == base or other_col.lower() == f"{base}s":
                        relationships.append(
                            InferredRelationship(
                                source_column=col,
                                target_column=other_col,
                                relationship_type="foreign_key",
                                description=f"{col} likely references {other_col}",
                                confidence=0.6,
                            )
                        )

        # Check for derived columns (e.g., total = price * quantity)
        numeric_cols = sample.select_dtypes(include=[np.number]).columns.tolist()
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1 :]:
                try:
                    corr = sample[col1].corr(sample[col2])
                    if abs(corr) > 0.9:
                        relationships.append(
                            InferredRelationship(
                                source_column=col1,
                                target_column=col2,
                                relationship_type="correlated",
                                description=f"Highly correlated (r={corr:.2f})",
                                confidence=0.8,
                            )
                        )
                except Exception:
                    pass

        return relationships

    def _infer_domain(self, columns: Dict[str, Dict[str, Any]]) -> str:
        """Infer data domain from column types."""
        types = [c.get("inferred_type", "") for c in columns.values()]
        descriptions = " ".join(c.get("description", "") for c in columns.values()).lower()

        if (
            any(t in ["diagnosis", "patient", "medical"] for t in types)
            or "medical" in descriptions
            or "patient" in descriptions
        ):
            return "healthcare"

        if (
            any(t in ["price", "amount", "currency"] for t in types)
            or "price" in descriptions
            or "transaction" in descriptions
        ):
            return "finance"

        if "product" in descriptions or "order" in descriptions:
            return "retail"

        if "user" in descriptions or "customer" in descriptions:
            return "customer"

        return "general"

    def _suggest_method(self, sample: pd.DataFrame, columns: Dict[str, Dict[str, Any]]) -> str:
        """Suggest generation method based on data characteristics."""
        n_rows = len(sample)
        n_cols = len(sample.columns)
        n_numeric = len(sample.select_dtypes(include=[np.number]).columns)
        n_categorical = len(sample.select_dtypes(include=["object", "category"]).columns)

        # Small dataset or mostly numeric: Gaussian Copula
        if n_rows < 1000 or n_numeric > n_cols * 0.8:
            return "gaussian_copula"

        # Large dataset with mixed types: CTGAN
        if n_rows > 10000 and n_categorical > 0:
            return "ctgan"

        # Medium dataset: TVAE
        if n_rows > 1000:
            return "tvae"

        return "gaussian_copula"


def infer_schema(
    data: pd.DataFrame,
    llm_provider: Optional[str] = None,
    api_key: Optional[str] = None,
    verbose: bool = True,
) -> InferredSchema:
    """Convenience function to infer schema from data.

    Args:
        data: Input DataFrame
        llm_provider: "openai", "anthropic", or None
        api_key: API key for LLM
        verbose: Print progress

    Returns:
        InferredSchema
    """
    inferrer = LLMSchemaInferrer(
        llm_provider=llm_provider,
        api_key=api_key,
    )

    if verbose:
        print(f"Inferring schema for {len(data.columns)} columns...")

    schema = inferrer.infer(data, use_llm=llm_provider is not None)

    if verbose:
        print(f"Detected domain: {schema.data_domain}")
        print(f"Suggested method: {schema.suggested_generation_method}")
        print(f"Found {len(schema.constraints)} constraints")
        print(f"Found {len(schema.relationships)} relationships")

    return schema
