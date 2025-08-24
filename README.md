# Efficient Table Error Detection with LLMs

This project leverages Large Language Models (LLMs) for efficient table error detection. It supports multiple datasets and models, including data preprocessing, feature extraction, error detection, and evaluation workflows.
<img width="819" height="384" alt="framework" src="https://github.com/user-attachments/assets/c0b1ab6a-fd67-42a0-bce4-c63d52fcc70c" />

## Directory Structure

```
├── config/           # Configuration files (models, prompts, etc.)
├── datasets/         # Table datasets (clean/dirty/rule)
├── experiments/      # Experiment scripts
├── src/              # Core code
│   ├── data/         # Data loading and preprocessing
│   ├── detection/    # Error detection and feature extraction
│   ├── evaluation/   # Evaluation and metrics
│   └── llm/          # LLM integration and management
├── tests/            # Unit tests
├── requirements.txt  # Dependency list
├── setup.py          # Installation script
└── README.md         # Project documentation
```

## Installation

Python 3.10 or above is recommended.

```bash
pip install -r requirements.txt
```

Or install via setup.py:

```bash
python setup.py install
```

## Quick Start

For example, to run the main detection script on the beers dataset:

```bash
python experiments/llm_ed.py --dataset beers
```

For more parameters and usage, please refer to the script comments.

## Dependencies

- transformers
- openai
- pandas
- numpy
- pyyaml

See requirements.txt for the full list.


## License

MIT
