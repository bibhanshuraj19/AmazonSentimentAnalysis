# Amazon Reviews Sentiment Analysis

A comprehensive sentiment analysis project that compares different approaches to analyze Amazon product reviews using both traditional NLTK VADER sentiment analyzer and modern transformer-based models.

## Overview

This project analyzes Amazon product reviews to understand customer sentiment using multiple sentiment analysis techniques. It compares the performance of VADER (Valence Aware Dictionary and sEntiment Reasoner) with RoBERTa-based transformer models to provide insights into customer opinions.

## Features

- **Data Processing**: Loads and processes Amazon reviews dataset
- **NLTK Text Processing**: 
  - Text tokenization
  - Part-of-speech tagging
  - Named entity recognition
- **Dual Sentiment Analysis**:
  - VADER sentiment analysis (rule-based)
  - RoBERTa transformer model (deep learning-based)
- **Comparative Analysis**: Side-by-side comparison of both sentiment analysis approaches
- **Data Visualization**: 
  - Review distribution by star ratings
  - Sentiment scores visualization
  - Model comparison plots

## Project Structure

```
.
├── amazon.ipynb           # Main analysis notebook
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .gitignore             # Git ignore rules
└── data/                  # Place datasets here (ignored by Git)
    └── .gitkeep           # Keeps the folder in Git
```

## Dependencies

```python
pandas
numpy
seaborn
matplotlib
nltk
transformers
scipy
tqdm
```

## Dataset

The project uses an Amazon Reviews dataset (`Reviews.csv`) containing:
- Review text
- Star ratings (1-5 stars)
- Product metadata
- Review IDs

## Methodology

### 1. VADER Sentiment Analysis
- Uses NLTK's SentimentIntensityAnalyzer
- Provides compound, positive, negative, and neutral scores
- Rule-based approach with lexicon and grammatical heuristics

### 2. RoBERTa Transformer Model
- Uses pre-trained `cardiffnlp/twitter-roberta-base-sentiment` model
- Provides probability scores for negative, neutral, and positive sentiments
- Deep learning approach trained on Twitter data

### 3. Comparison Analysis
- Correlates sentiment scores with actual star ratings
- Identifies discrepancies between models
- Analyzes edge cases (e.g., negative sentiment in 5-star reviews)

## Key Insights

The analysis reveals:
- How well sentiment scores correlate with star ratings
- Differences in performance between VADER and RoBERTa models
- Edge cases where sentiment doesn't match star ratings
- Comparative strengths of rule-based vs. transformer-based approaches

## macOS Quickstart

1. Create and activate a virtual environment:
   ```bash
   cd path/to/repo
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the dataset:
   - Save `Reviews.csv` to `data/Reviews.csv`, or update the notebook to point to your file:
   ```python
   DATA_PATH = "data/Reviews.csv"
   df = pd.read_csv(DATA_PATH)
   ```
4. Start Jupyter and select the environment kernel:
   ```bash
   python -m ipykernel install --user --name amazon-sentiment --display-name "Python (amazon-sentiment)"
   jupyter notebook
   ```
5. Open `amazon.ipynb` and run the cells. The first run will download required NLTK and model data to `~/nltk_data` and `~/.cache/huggingface`.

Notes for Apple Silicon (M1/M2/M3):
- The provided `requirements.txt` installs CPU-only PyTorch wheels that work on macOS.
- If you hit NLTK data permission issues, set `export NLTK_DATA="$HOME/nltk_data"` and re-run the download cells.

## Usage

1. **Setup Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Download NLTK Data**:
   The notebook automatically downloads required NLTK packages:
   - averaged_perceptron_tagger_eng
   - maxent_ne_chunker_tab
   - words
   - vader_lexicon
   - punkt_tab

3. **Run Analysis**:
   Execute the Jupyter notebook cells sequentially to:
   - Load and explore the dataset
   - Perform VADER sentiment analysis
   - Run RoBERTa model predictions
   - Compare results and visualize findings

## Results

The project generates:
- Sentiment score distributions by star rating
- Comparative visualizations between VADER and RoBERTa
- Identification of reviews where sentiment and ratings diverge
- Statistical analysis of model performance

## Model Performance

Both models are evaluated on their ability to:
- Correctly identify sentiment polarity
- Correlate with user-provided star ratings
- Handle nuanced language and context
- Process different review lengths and styles

## Future Enhancements

Potential improvements could include:
- Testing additional transformer models (BERT, DistilBERT)
- Fine-tuning models on Amazon review data
- Implementing ensemble methods
- Adding aspect-based sentiment analysis
- Expanding to multi-language reviews

## Notes

- The analysis uses a subset of 500 reviews for computational efficiency
- Some reviews may cause runtime errors during processing (handled gracefully)
- The project demonstrates the evolution from traditional NLP to modern transformer approaches

## Author

This project demonstrates practical applications of sentiment analysis techniques for e-commerce review analysis, showcasing both traditional and modern NLP approaches.
