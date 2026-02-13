# Text-to-Python Code Generation using Seq2Seq Models

This project implements and compares three sequence-to-sequence models for translating English docstrings into Python code.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/SOUMITRAPAUL/seq2seq-main
cd seq2seq
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python3 -c "import nltk; nltk.download('punkt')"

# Prepare data
python3 -c "from data.preprocess import prepare_data; prepare_data()"

# Train all models
python3 train.py --model all --num_epochs 2

# Evaluate
python3 evaluate.py --model all

# Visualize attention
python3 visualize_attention.py --num_examples 5
```

## Overview

Here’s a crisp 3-line comparison based on your average BLEU-4 scores:

* **Vanilla RNN**: Baseline performance with BLEU-4 ≈ **0.0314**, struggles with long-term dependencies.
* **LSTM Seq2Seq**: Improved context handling, BLEU-4 ≈ **0.0345**, slightly better than RNN.
* **LSTM + Attention**: Best performance, BLEU-4 ≈ **0.0419**, dynamically focuses on relevant tokens for accurate code generation.


## Usage

**Note:** All commands should be run from the repository root directory.

### Step 1: Prepare Data

Download and preprocess the CodeSearchNet Python dataset:

```bash
python3 -c "from data.preprocess import prepare_data; prepare_data()"
```

This will:
- Download the CodeSearchNet dataset from Hugging Face
- Filter samples by length constraints (docstring ≤ 50 tokens, code ≤ 80 tokens)
- Create train/val/test splits (8000/1000/1000 samples)
- Build vocabularies
- Save processed data to `./data/`

### Step 2: Train Models

Train all three models:

```bash
python3 train.py --model all --num_epochs 2
```

Or train individual models:

```bash
# Train only RNN
python3 train.py --model rnn --num_epochs 2

# Train only LSTM
python3 train.py --model lstm --num_epochs 2

# Train only Attention
python3 train.py --model attention --num_epochs 2
```

**Training Options:**
- `--embed_dim`: Embedding dimension (default: 256)
- `--hidden_dim`: Hidden state dimension (default: 256)
- `--num_layers`: Number of RNN/LSTM layers (default: 1)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--teacher_forcing_ratio`: Teacher forcing probability (default: 0.5)
- `--num_epochs`: Number of training epochs (default: 2)

### Step 3: Evaluate Models

Evaluate all trained models on the test set:

```bash
python3 evaluate.py --model all
```

This computes:
- **Token-level accuracy**: Proportion of correctly predicted tokens
- **Exact match accuracy**: Proportion of perfectly matched sequences
- **BLEU scores**: BLEU-1, BLEU-2, BLEU-3, BLEU-4, and corpus BLEU
- **Performance by sequence length**: Analysis across different length ranges
- **Performance by docstring length**: Analysis by source sequence length

### Step 4: Visualize Attention

Generate attention heatmaps for the LSTM with Attention model:

```bash
python3 visualize_attention.py --num_examples 5
```

This creates:
- Attention heatmaps showing alignment between source and target tokens
- Analysis of attention patterns
- Interpretation of which source tokens influence target generation

### Step 5: Error Analysis (Optional)

Run detailed error analysis:

```bash
python3 error_analysis.py --model all
```

## Model Architectures

* **RNN Seq2Seq**: Simple encoder/decoder, fixed-length context, struggles with long sequences.
* **LSTM Seq2Seq**: Encoder/decoder LSTM, better long-term dependencies via gating.
* **LSTM + Attention**: Bidirectional LSTM encoder, attention decoder, dynamic context, best performance.

## Configuration

* Embedding & hidden dim: 256
* Optimizer: Adam (lr=0.001)
* Loss: CrossEntropyLoss (ignore padding)
* Teacher forcing: 0.5, Gradient clipping: 1.0

## Dataset

* **Source**: CodeSearchNet Python ([Hugging Face](https://huggingface.co/datasets/Nan-Do/code-search-net-python))
* **Input**: Docstrings → **Output**: Python code
* **Samples**: Train 8k / Val 1k / Test 1k
* **Preprocessing**: Tokenize, lowercase docstrings, special tokens `<sos> <eos> <pad> <unk>`

## Results

Saved in `./results/`:

* Training & evaluation JSON
* Plots: loss curves & attention heatmaps
* Example predictions & attention analysis

## Evaluation Metrics

* **Token Accuracy**: Correct token %
* **Exact Match**: Fully matched sequences %
* **BLEU**: BLEU-1 → BLEU-4, corpus BLEU

## Error Analysis

* Performance vs sequence length
* Common errors: syntax, operators, indentation, variable names, incomplete logic

## Attention Visualization

* X-axis: source tokens, Y-axis: generated code
* Color = attention weight
* Shows semantic alignment (e.g., `"maximum"` → `max()`)

## Tips

* **Slow training**: use GPU or reduce dataset size
* **Module errors**: activate venv, run in repo root, `pip install -r requirements.txt`

## Citation

```
@misc{codesearchnet,
  title={CodeSearchNet Challenge},
  author={Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
  year={2019}
}
```
