import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from typing import List, Dict, Optional
import random

class ReviewScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def scrape_amazon_reviews(self, product_url: str, num_pages: int = 5) -> pd.DataFrame:
        """
        Scrape reviews from Amazon product pages
        
        Args:
            product_url: URL of the Amazon product
            num_pages: Number of review pages to scrape
            
        Returns:
            DataFrame containing reviews
        """
        reviews = []
        
        for page in range(1, num_pages + 1):
            try:
                # Modify URL to get reviews page
                review_url = product_url.replace('/dp/', '/product-reviews/') + f'?pageNumber={page}'
                response = requests.get(review_url, headers=self.headers)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract reviews
                review_elements = soup.find_all('div', {'data-hook': 'review'})
                
                for review in review_elements:
                    try:
                        rating = review.find('i', {'data-hook': 'review-star-rating'}).text.split('.')[0]
                        title = review.find('a', {'data-hook': 'review-title'}).text.strip()
                        body = review.find('span', {'data-hook': 'review-body'}).text.strip()
                        date = review.find('span', {'data-hook': 'review-date'}).text.strip()
                        
                        reviews.append({
                            'rating': int(rating),
                            'title': title,
                            'body': body,
                            'date': date,
                            'source': 'amazon'
                        })
                    except AttributeError:
                        continue
                
                # Random delay to avoid being blocked
                time.sleep(random.uniform(2, 5))
                
            except Exception as e:
                print(f"Error scraping page {page}: {str(e)}")
                continue
                
        return pd.DataFrame(reviews)
    
    def scrape_yelp_reviews(self, business_url: str, num_pages: int = 5) -> pd.DataFrame:
        """
        Scrape reviews from Yelp business pages
        
        Args:
            business_url: URL of the Yelp business
            num_pages: Number of review pages to scrape
            
        Returns:
            DataFrame containing reviews
        """
        reviews = []
        
        for page in range(0, num_pages):
            try:
                # Modify URL to get reviews page
                review_url = f"{business_url}?start={page*10}"
                response = requests.get(review_url, headers=self.headers)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract reviews
                review_elements = soup.find_all('div', {'class': 'review'})
                
                for review in review_elements:
                    try:
                        rating = review.find('div', {'class': 'rating'}).get('aria-label', '').split()[0]
                        title = review.find('h3', {'class': 'review-title'}).text.strip()
                        body = review.find('p', {'class': 'review-content'}).text.strip()
                        date = review.find('span', {'class': 'rating-qualifier'}).text.strip()
                        
                        reviews.append({
                            'rating': float(rating),
                            'title': title,
                            'body': body,
                            'date': date,
                            'source': 'yelp'
                        })
                    except AttributeError:
                        continue
                
                # Random delay to avoid being blocked
                time.sleep(random.uniform(2, 5))
                
            except Exception as e:
                print(f"Error scraping page {page}: {str(e)}")
                continue
                
        return pd.DataFrame(reviews)
    
    def save_reviews(self, reviews_df: pd.DataFrame, filename: str):
        """
        Save scraped reviews to CSV file
        
        Args:
            reviews_df: DataFrame containing reviews
            filename: Name of the file to save
        """
        reviews_df.to_csv(f'data/{filename}.csv', index=False)
        
    def load_reviews(self, filename: str) -> pd.DataFrame:
        """
        Load reviews from CSV file
        
        Args:
            filename: Name of the file to load
            
        Returns:
            DataFrame containing reviews
        """
        return pd.read_csv(f'data/{filename}.csv') 