import pandas as pd
import numpy as np
from gensim import corpora, models
import spacy
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pyLDAvis.gensim_models
import pyLDAvis

class TopicModeler:
    def __init__(self, n_topics: int = 5):
        """
        Initialize the topic modeler
        
        Args:
            n_topics: Number of topics to extract
        """
        self.n_topics = n_topics
        self.nlp = spacy.load('en_core_web_sm')
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text using SpaCy
        
        Args:
            text: Input text to preprocess
            
        Returns:
            List of preprocessed tokens
        """
        doc = self.nlp(text.lower())
        return [token.text for token in doc 
                if not token.is_stop 
                and not token.is_punct 
                and not token.is_space 
                and len(token.text) > 2]
    
    def prepare_corpus(self, texts: List[str]) -> Tuple[corpora.Dictionary, List[List[Tuple[int, int]]]]:
        """
        Prepare corpus for topic modeling
        
        Args:
            texts: List of texts to process
            
        Returns:
            Tuple of (dictionary, corpus)
        """
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Create dictionary
        self.dictionary = corpora.Dictionary(processed_texts)
        
        # Filter out rare and common words
        self.dictionary.filter_extremes(no_below=2, no_above=0.5)
        
        # Create corpus
        self.corpus = [self.dictionary.doc2bow(text) for text in processed_texts]
        
        return self.dictionary, self.corpus
    
    def train_lda(self, texts: List[str], n_topics: int = None) -> models.LdaModel:
        """
        Train LDA model
        
        Args:
            texts: List of texts to analyze
            n_topics: Number of topics (overrides initialization parameter)
            
        Returns:
            Trained LDA model
        """
        if n_topics is not None:
            self.n_topics = n_topics
            
        # Prepare corpus if not already done
        if self.dictionary is None or self.corpus is None:
            self.prepare_corpus(texts)
            
        # Train LDA model
        self.lda_model = models.LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.n_topics,
            random_state=42,
            passes=15
        )
        
        return self.lda_model
    
    def get_topic_keywords(self, n_words: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get keywords for each topic
        
        Args:
            n_words: Number of keywords to return per topic
            
        Returns:
            Dictionary mapping topic IDs to lists of (word, probability) tuples
        """
        if self.lda_model is None:
            raise ValueError("LDA model not trained yet")
            
        topics = {}
        for topic_id in range(self.n_topics):
            topics[topic_id] = self.lda_model.show_topic(topic_id, topn=n_words)
        return topics
    
    def get_document_topics(self, texts: List[str]) -> List[List[Tuple[int, float]]]:
        """
        Get topic distribution for each document
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of topic distributions for each document
        """
        if self.lda_model is None:
            raise ValueError("LDA model not trained yet")
            
        # Preprocess new texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        bow_corpus = [self.dictionary.doc2bow(text) for text in processed_texts]
        
        # Get topic distributions
        return [self.lda_model.get_document_topics(bow) for bow in bow_corpus]
    
    def visualize_topics(self, save_path: str = None):
        """
        Create interactive visualization of topics
        
        Args:
            save_path: Optional path to save the visualization
        """
        if self.lda_model is None:
            raise ValueError("LDA model not trained yet")
            
        vis_data = pyLDAvis.gensim_models.prepare(self.lda_model, self.corpus, self.dictionary)
        
        if save_path:
            pyLDAvis.save_html(vis_data, save_path)
            
        return vis_data
    
    def plot_topic_distribution(self, texts: List[str], save_path: str = None):
        """
        Plot topic distribution across documents
        
        Args:
            texts: List of texts to analyze
            save_path: Optional path to save the plot
        """
        if self.lda_model is None:
            raise ValueError("LDA model not trained yet")
            
        # Get topic distributions
        doc_topics = self.get_document_topics(texts)
        
        # Convert to numpy array for easier plotting
        topic_matrix = np.zeros((len(doc_topics), self.n_topics))
        for i, topics in enumerate(doc_topics):
            for topic_id, prob in topics:
                topic_matrix[i, topic_id] = prob
                
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(topic_matrix, cmap='YlOrRd')
        plt.title('Topic Distribution Across Documents')
        plt.xlabel('Topic ID')
        plt.ylabel('Document ID')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def analyze_customer_feedback(self, feedback_df: pd.DataFrame, text_column: str = 'body') -> Dict:
        """
        Analyze customer feedback using topic modeling
        
        Args:
            feedback_df: DataFrame containing customer feedback
            text_column: Name of the column containing feedback text
            
        Returns:
            Dictionary containing analysis results
        """
        # Train LDA model
        self.train_lda(feedback_df[text_column].tolist())
        
        # Get topic keywords
        topic_keywords = self.get_topic_keywords()
        
        # Get document topics
        doc_topics = self.get_document_topics(feedback_df[text_column].tolist())
        
        # Calculate topic frequencies
        topic_freq = defaultdict(int)
        for doc_topics in doc_topics:
            for topic_id, prob in doc_topics:
                topic_freq[topic_id] += 1
                
        # Sort topics by frequency
        sorted_topics = sorted(topic_freq.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'topic_keywords': topic_keywords,
            'document_topics': doc_topics,
            'topic_frequencies': dict(sorted_topics)
        } 