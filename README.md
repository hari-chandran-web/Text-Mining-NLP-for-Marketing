# Text Mining & NLP for Marketing

This project implements advanced text mining and Natural Language Processing (NLP) techniques for marketing analysis, focusing on two main components:

1. Sentiment Analysis of Product Reviews
2. Topic Modeling on Customer Feedback

## Project Structure

```
├── data/                   # Data storage directory
├── notebooks/             # Jupyter notebooks for analysis
├── src/                   # Source code
│   ├── sentiment/        # Sentiment analysis modules
│   └── topic_modeling/   # Topic modeling modules
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Features

### 1. Sentiment Analysis
- Scraping Amazon/Yelp reviews
- Multiple sentiment analysis approaches:
  - TextBlob
  - VADER
  - BERT-based models
  - Large Language Models
- Comparative analysis of different methods
- Visualization of results

### 2. Topic Modeling
- Latent Dirichlet Allocation (LDA) implementation
- Customer feedback analysis
- Topic visualization
- Actionable insights extraction

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
```

4. Download SpaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

1. For sentiment analysis:
   - Run the sentiment analysis notebook in `notebooks/sentiment_analysis.ipynb`
   - Use the scraping module to collect reviews
   - Compare different sentiment analysis methods

2. For topic modeling:
   - Run the topic modeling notebook in `notebooks/topic_modeling.ipynb`
   - Process customer feedback
   - Generate topic visualizations
