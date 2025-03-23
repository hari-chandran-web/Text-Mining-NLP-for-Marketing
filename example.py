from sentiment.review_scraper import ReviewScraper
from sentiment.sentiment_analyzer import SentimentAnalyzer
from topic_modeling.topic_modeler import TopicModeler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Initialize components
    scraper = ReviewScraper()
    sentiment_analyzer = SentimentAnalyzer()
    topic_modeler = TopicModeler(n_topics=5)
    
    # Example Amazon product URL (replace with actual URL)
    amazon_url = "https://www.amazon.com/product-url"
    
    # 1. Scrape reviews
    print("Scraping reviews...")
    reviews = scraper.scrape_amazon_reviews(amazon_url, num_pages=5)
    scraper.save_reviews(reviews, 'amazon_reviews')
    
    # 2. Analyze sentiment
    print("\nAnalyzing sentiment...")
    sentiment_results = sentiment_analyzer.analyze_reviews(reviews)
    
    # 3. Perform topic modeling
    print("\nPerforming topic modeling...")
    topic_results = topic_modeler.analyze_customer_feedback(reviews)
    
    # 4. Generate visualizations
    print("\nGenerating visualizations...")
    
    # Sentiment distribution
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sns.histplot(data=sentiment_results, x='textblob_polarity', bins=30)
    plt.title('TextBlob Polarity Distribution')
    
    plt.subplot(1, 3, 2)
    sns.histplot(data=sentiment_results, x='vader_compound', bins=30)
    plt.title('VADER Compound Distribution')
    
    plt.subplot(1, 3, 3)
    sns.histplot(data=sentiment_results, x='bert_score', bins=30)
    plt.title('BERT Score Distribution')
    
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png')
    plt.close()
    
    # Topic distribution
    topic_modeler.plot_topic_distribution(reviews['body'].tolist(), 'topic_distribution.png')
    
    # Topic visualization
    topic_modeler.visualize_topics('topic_visualization.html')
    
    # 5. Print summary statistics
    print("\nSummary Statistics:")
    print("\nSentiment Analysis:")
    print(sentiment_analyzer.compare_methods(sentiment_results))
    
    print("\nTopic Modeling:")
    print("\nTop Keywords for Each Topic:")
    for topic_id, keywords in topic_results['topic_keywords'].items():
        print(f"\nTopic {topic_id}:")
        for word, prob in keywords:
            print(f"  {word}: {prob:.4f}")
    
    print("\nTopic Frequencies:")
    for topic_id, freq in topic_results['topic_frequencies'].items():
        print(f"Topic {topic_id}: {freq} documents")

if __name__ == "__main__":
    main() 