import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from typing import List, Dict, Union
import torch

class SentimentAnalyzer:
    def __init__(self):
        """Initialize sentiment analyzers"""
        self.vader = SentimentIntensityAnalyzer()
        self.bert_sentiment = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=0 if torch.cuda.is_available() else -1
        )
        
    def analyze_textblob(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing polarity and subjectivity scores
        """
        analysis = TextBlob(text)
        return {
            'polarity': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity
        }
    
    def analyze_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing VADER sentiment scores
        """
        return self.vader.polarity_scores(text)
    
    def analyze_bert(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyze sentiment using BERT
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing BERT sentiment prediction and score
        """
        result = self.bert_sentiment(text)[0]
        return {
            'label': result['label'],
            'score': result['score']
        }
    
    def analyze_reviews(self, reviews_df: pd.DataFrame, methods: List[str] = ['textblob', 'vader', 'bert']) -> pd.DataFrame:
        """
        Analyze sentiment for all reviews using specified methods
        
        Args:
            reviews_df: DataFrame containing reviews
            methods: List of sentiment analysis methods to use
            
        Returns:
            DataFrame with added sentiment analysis columns
        """
        results_df = reviews_df.copy()
        
        for method in methods:
            print(f"Analyzing sentiment using {method}...")
            
            if method == 'textblob':
                results_df['textblob_polarity'] = results_df['body'].apply(
                    lambda x: self.analyze_textblob(x)['polarity']
                )
                results_df['textblob_subjectivity'] = results_df['body'].apply(
                    lambda x: self.analyze_textblob(x)['subjectivity']
                )
                
            elif method == 'vader':
                results_df['vader_compound'] = results_df['body'].apply(
                    lambda x: self.analyze_vader(x)['compound']
                )
                results_df['vader_pos'] = results_df['body'].apply(
                    lambda x: self.analyze_vader(x)['pos']
                )
                results_df['vader_neg'] = results_df['body'].apply(
                    lambda x: self.analyze_vader(x)['neg']
                )
                results_df['vader_neu'] = results_df['body'].apply(
                    lambda x: self.analyze_vader(x)['neu']
                )
                
            elif method == 'bert':
                results_df['bert_sentiment'] = results_df['body'].apply(
                    lambda x: self.analyze_bert(x)['label']
                )
                results_df['bert_score'] = results_df['body'].apply(
                    lambda x: self.analyze_bert(x)['score']
                )
        
        return results_df
    
    def compare_methods(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare results from different sentiment analysis methods
        
        Args:
            results_df: DataFrame containing results from multiple methods
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison = pd.DataFrame()
        
        # Calculate correlation between different methods
        if 'textblob_polarity' in results_df.columns and 'vader_compound' in results_df.columns:
            comparison['textblob_vader_corr'] = results_df['textblob_polarity'].corr(results_df['vader_compound'])
            
        if 'textblob_polarity' in results_df.columns and 'bert_score' in results_df.columns:
            comparison['textblob_bert_corr'] = results_df['textblob_polarity'].corr(results_df['bert_score'])
            
        if 'vader_compound' in results_df.columns and 'bert_score' in results_df.columns:
            comparison['vader_bert_corr'] = results_df['vader_compound'].corr(results_df['bert_score'])
        
        # Calculate agreement rates
        if all(col in results_df.columns for col in ['textblob_polarity', 'vader_compound', 'bert_score']):
            comparison['all_methods_agreement'] = (
                (results_df['textblob_polarity'] > 0) == 
                (results_df['vader_compound'] > 0) == 
                (results_df['bert_score'] > 0.5)
            ).mean()
        
        return comparison 