---
sidebar_position: 3
title: CLI Reference
---

# CLI Reference

Complete reference for the Genesis command-line interface.

## Installation

```bash
pip install genesis-synth
```

The CLI is available as `genesis` after installation.

## Global Options

```bash
genesis [OPTIONS] COMMAND [ARGS]

Options:
  --version       Show version and exit
  --verbose, -v   Enable verbose output
  --quiet, -q     Suppress output
  --config FILE   Path to config file
  --help          Show help message
```

---

## Commands

### generate

Generate synthetic data from a trained model or directly from data.

```bash
genesis generate [OPTIONS] INPUT OUTPUT
```

**Arguments:**
- `INPUT`: Path to input CSV/Parquet file or trained model
- `OUTPUT`: Path for output synthetic data

**Options:**
```bash
--samples, -n INTEGER      Number of samples to generate [default: same as input]
--method TEXT              Generation method (ctgan, tvae, gaussian_copula)
--discrete-columns TEXT    Comma-separated list of categorical columns
--epochs INTEGER           Training epochs [default: 300]
--batch-size INTEGER       Training batch size [default: 500]
--seed INTEGER             Random seed for reproducibility
--format TEXT              Output format (csv, parquet) [default: csv]
```

**Examples:**
```bash
# Basic generation
genesis generate customers.csv synthetic.csv --samples 10000

# With options
genesis generate customers.csv synthetic.csv \
  --samples 10000 \
  --method ctgan \
  --discrete-columns status,region,category \
  --epochs 500
```

---

### automl

Use AutoML to automatically select the best method.

```bash
genesis automl [OPTIONS] INPUT OUTPUT
```

**Options:**
```bash
--samples, -n INTEGER      Number of samples [default: same as input]
--mode TEXT                Mode: fast, balanced, quality [default: balanced]
--discrete-columns TEXT    Categorical columns (comma-separated)
--output-report TEXT       Path for quality report
```

**Examples:**
```bash
# Quick AutoML generation
genesis automl data.csv synthetic.csv

# High quality mode with report
genesis automl data.csv synthetic.csv \
  --mode quality \
  --samples 50000 \
  --output-report report.html
```

---

### evaluate

Evaluate synthetic data quality.

```bash
genesis evaluate [OPTIONS] REAL SYNTHETIC
```

**Arguments:**
- `REAL`: Path to real data
- `SYNTHETIC`: Path to synthetic data

**Options:**
```bash
--target TEXT              Target column for ML utility evaluation
--output TEXT              Output path for report
--format TEXT              Report format: text, json, html [default: text]
```

**Examples:**
```bash
# Basic evaluation
genesis evaluate real.csv synthetic.csv

# With ML utility and HTML report
genesis evaluate real.csv synthetic.csv \
  --target churn \
  --output report.html \
  --format html
```

---

### privacy-audit

Run privacy attack tests on synthetic data.

```bash
genesis privacy-audit [OPTIONS] REAL SYNTHETIC
```

**Options:**
```bash
--sensitive-columns TEXT   Sensitive columns (comma-separated)
--quasi-identifiers TEXT   Quasi-identifier columns (comma-separated)
--output TEXT              Output path for report
--threshold FLOAT          Privacy score threshold [default: 0.9]
```

**Examples:**
```bash
# Basic audit
genesis privacy-audit real.csv synthetic.csv

# With specific columns
genesis privacy-audit real.csv synthetic.csv \
  --sensitive-columns income,health_status \
  --quasi-identifiers age,gender,zip_code \
  --output privacy_report.html
```

---

### drift

Detect statistical drift between datasets.

```bash
genesis drift [OPTIONS] BASELINE CURRENT
```

**Options:**
```bash
--threshold FLOAT          Drift threshold [default: 0.1]
--columns TEXT             Columns to check (comma-separated, default: all)
--output TEXT              Output path for report
--format TEXT              Output format: text, json, html [default: text]
```

**Examples:**
```bash
# Check drift
genesis drift training.csv production.csv

# With specific columns and threshold
genesis drift training.csv production.csv \
  --threshold 0.05 \
  --columns age,income,status \
  --format json
```

---

### augment

Balance an imbalanced dataset.

```bash
genesis augment [OPTIONS] INPUT OUTPUT
```

**Options:**
```bash
--target TEXT              Target column to balance [required]
--ratio FLOAT              Target ratio minority/majority [default: 1.0]
--strategy TEXT            Strategy: oversample, undersample, hybrid [default: oversample]
--discrete-columns TEXT    Categorical columns (comma-separated)
```

**Examples:**
```bash
# Balance dataset
genesis augment imbalanced.csv balanced.csv --target fraud

# Partial balance
genesis augment imbalanced.csv balanced.csv \
  --target churn \
  --ratio 0.5 \
  --strategy hybrid
```

---

### version

Manage dataset versions.

```bash
genesis version [SUBCOMMAND] [OPTIONS]
```

**Subcommands:**

#### init
```bash
genesis version init [PATH]
```
Initialize a dataset repository.

#### save
```bash
genesis version save [OPTIONS] FILE
```
Save a dataset version.

Options:
- `--message, -m TEXT`: Version message
- `--tag, -t TEXT`: Tags (can be used multiple times)

#### list
```bash
genesis version list [OPTIONS]
```
List dataset versions.

Options:
- `--tag TEXT`: Filter by tag

#### load
```bash
genesis version load [OPTIONS] VERSION_ID OUTPUT
```
Load a dataset version.

#### compare
```bash
genesis version compare VERSION1 VERSION2
```
Compare two versions.

#### tag
```bash
genesis version tag VERSION_ID TAG
```
Add tag to a version.

**Examples:**
```bash
# Initialize repository
genesis version init ./datasets

# Save with message and tags
genesis version save data.csv -m "Initial version" -t v1.0 -t production

# List versions
genesis version list
genesis version list --tag production

# Compare versions
genesis version compare abc123 def456

# Load a version
genesis version load abc123 output.csv
genesis version load --tag production output.csv
```

---

### domain

Generate domain-specific data.

```bash
genesis domain [TYPE] [OPTIONS]
```

**Types:**
- `names`: Personal names
- `emails`: Email addresses
- `phones`: Phone numbers
- `addresses`: Street addresses
- `composite`: Combined records

**Options:**
```bash
--count, -n INTEGER        Number of records [default: 100]
--locale TEXT              Locale code [default: en_US]
--output, -o TEXT          Output file path
--format TEXT              Output format: csv, json [default: csv]
```

**Examples:**
```bash
# Generate names
genesis domain names --count 1000 --locale en_US --output names.csv

# Generate emails
genesis domain emails --count 1000 --output emails.csv

# Generate addresses
genesis domain addresses --count 1000 --locale en_GB --output uk_addresses.csv

# Generate composite records
genesis domain composite \
  --config composite.yaml \
  --count 10000 \
  --output people.csv
```

---

### pipeline

Run data generation pipelines.

```bash
genesis pipeline [SUBCOMMAND] [OPTIONS]
```

**Subcommands:**

#### run
```bash
genesis pipeline run PIPELINE_FILE [OPTIONS]
```
Run a pipeline from YAML file.

Options:
- `--set KEY=VALUE`: Override pipeline settings

#### validate
```bash
genesis pipeline validate PIPELINE_FILE
```
Validate a pipeline without running.

#### visualize
```bash
genesis pipeline visualize PIPELINE_FILE
```
Show pipeline structure.

**Examples:**
```bash
# Run pipeline
genesis pipeline run pipeline.yaml

# Run with overrides
genesis pipeline run pipeline.yaml \
  --set steps.generate.n_samples=5000 \
  --set output=custom_output.csv

# Validate pipeline
genesis pipeline validate pipeline.yaml
```

---

### train

Train and save a generator model.

```bash
genesis train [OPTIONS] INPUT MODEL_OUTPUT
```

**Options:**
```bash
--method TEXT              Generation method [default: ctgan]
--discrete-columns TEXT    Categorical columns
--epochs INTEGER           Training epochs [default: 300]
--batch-size INTEGER       Batch size [default: 500]
--config TEXT              Path to config file
```

**Examples:**
```bash
# Train and save model
genesis train customers.csv customer_model.pkl \
  --method ctgan \
  --discrete-columns status,region \
  --epochs 500

# Load and generate later
genesis generate customer_model.pkl synthetic.csv --samples 10000
```

---

### info

Show information about Genesis installation.

```bash
genesis info
```

Output includes:
- Version
- Installed dependencies
- Available methods
- GPU availability
- Configuration paths

---

## Configuration File

Create `genesis.yaml` for default settings:

```yaml
defaults:
  method: ctgan
  epochs: 300
  batch_size: 500

privacy:
  epsilon: 1.0
  k: 5

output:
  format: csv
```

Use with:
```bash
genesis --config genesis.yaml generate data.csv output.csv
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | File not found |
| 4 | Generation error |
| 5 | Validation error |
| 6 | Privacy threshold not met |

---

## Shell Completion

### Bash
```bash
eval "$(_GENESIS_COMPLETE=bash_source genesis)"
```

### Zsh
```bash
eval "$(_GENESIS_COMPLETE=zsh_source genesis)"
```

### Fish
```bash
eval (env _GENESIS_COMPLETE=fish_source genesis)
```

Add to your shell config for persistent completion.
