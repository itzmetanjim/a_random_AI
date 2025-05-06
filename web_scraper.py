#!/usr/bin/env python
# Web Scraper for Dataset Creation
import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import random
import argparse

class WebScraper:
    def __init__(self, base_urls=None, output_dir="dataset"):
        """
        Initialize the web scraper
        
        Args:
            base_urls (list): List of URLs to scrape
            output_dir (str): Directory to save scraped data
        """
        self.base_urls = base_urls or []
        self.output_dir = output_dir
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def add_url(self, url):
        """Add a URL to the list of URLs to scrape"""
        self.base_urls.append(url)
        
    def fetch_page(self, url):
        """
        Fetch a webpage and return its content
        
        Args:
            url (str): URL to fetch
            
        Returns:
            BeautifulSoup object or None if request fails
        """
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
            
    def extract_text_content(self, soup, content_tags=None):
        """
        Extract text content from specific tags
        
        Args:
            soup (BeautifulSoup): Parsed HTML
            content_tags (dict): Dictionary mapping tag names to attributes
                                e.g. {'div': {'class': 'content'}}
        
        Returns:
            dict: Extracted text content
        """
        content = {}
        
        # Default tags to extract if none provided
        if content_tags is None:
            content_tags = {
                'h1': {},
                'h2': {},
                'h3': {},
                'p': {},
                'article': {},
                'div': {'class': ['content', 'article', 'main', 'post']}
            }
        
        # Extract text from specified tags
        for tag, attrs in content_tags.items():
            elements = soup.find_all(tag, attrs) if attrs else soup.find_all(tag)
            tag_content = [elem.get_text(strip=True) for elem in elements if elem.get_text(strip=True)]
            if tag_content:
                content[tag] = tag_content
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag and title_tag.text:
            content['title'] = title_tag.text
            
        return content
    
    def extract_links(self, soup, base_url):
        """
        Extract content links from the page, filtering out navigation/utility links
        
        Args:
            soup (BeautifulSoup): Parsed HTML
            base_url (str): Base URL to resolve relative links
            
        Returns:
            list: List of filtered content URLs
        """
        links = []
        # Get domain from base_url
        domain = None
        if "://" in base_url:
            domain = base_url.split("://")[1].split("/")[0]

        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            # Skip unwanted links
            if any(x in href for x in [
                'Special:', 'Talk:', 'User:', 'Template:', 'Category:', 
                'Help:', 'Portal:', 'File:', 'action=edit', 'oldid=', 
                'printable=yes', '#', 'action=history', '?title=',
                'index.php', 'Main_Page', 'login', 'signup', 'account',
                'diff=', 'curid=', 'veaction=', 'search'
            ]):
                continue
                
            # Handle relative URLs correctly
            if href.startswith('/'):
                # Remove trailing slash from base_url if it exists
                base = base_url.split('/')[0] + '//' + base_url.split('/')[2]
                href = f"{base}{href}"
            elif not href.startswith(('http://', 'https://')):
                # Skip non-http links like javascript:, mailto:, etc.
                continue
                
            # Make sure we're staying within the same domain
            if domain and domain not in href:
                continue
                
            # Avoid duplicate slashes that create invalid URLs
            href = re.sub(r'([^:])//+', r'\1/', href)
            
            links.append(href)
        
        return links
    
    def clean_text(self, text):
        """Clean extracted text"""
        if not text:
            return ""
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove non-alphanumeric characters except punctuation
        text = re.sub(r'[^\w\s.,!?:;\'"-]', '', text)
        
        return text
        
    def create_dataset(self, content_tags=None, max_pages=50, delay=(1, 3)):
        """
        Create a dataset from scraped URLs
        
        Args:
            content_tags (dict): Dictionary mapping tag names to attributes
            max_pages (int): Maximum number of pages to scrape
            delay (tuple): Random delay range between requests (min, max) in seconds
            
        Returns:
            pd.DataFrame: DataFrame with extracted content
        """
        all_data = []
        visited_urls = set()
        urls_to_visit = self.base_urls.copy()
        
        page_count = 0
        
        while urls_to_visit and page_count < max_pages:
            url = urls_to_visit.pop(0)
            
            # Skip if already visited
            if url in visited_urls:
                continue
                
            print(f"Scraping {url} ({page_count + 1}/{max_pages})")
            visited_urls.add(url)
            
            # Fetch and parse page
            soup = self.fetch_page(url)
            if not soup:
                continue
                
            # Extract content
            content = self.extract_text_content(soup, content_tags)
            
            # Combine and clean all text
            all_text = ""
            for tag_type, text_list in content.items():
                if tag_type == 'title':
                    all_text += f"TITLE: {text_list}\n\n"
                else:
                    all_text += "\n".join(text_list) + "\n\n"
            
            # Create data entry
            data_entry = {
                'url': url,
                'title': content.get('title', ''),
                'content': self.clean_text(all_text),
                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            all_data.append(data_entry)
            page_count += 1
            
            # Extract new links to visit
            new_links = self.extract_links(soup, url)
            for link in new_links:
                if link not in visited_urls and link not in urls_to_visit:
                    for base_url in self.base_urls:
                        # Only add URLs from the same domain(s)
                        if link.startswith(base_url):
                            urls_to_visit.append(link)
                            break
            
            # Random delay to be respectful
            time.sleep(random.uniform(delay[0], delay[1]))
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        # Save dataset
        csv_path = os.path.join(self.output_dir, 'scraped_data.csv')
        df.to_csv(csv_path, index=False)
        print(f"Dataset saved to {csv_path}")
        
        return df
        
    def save_text_files(self, df):
        """
        Save individual text files for each entry
        
        Args:
            df (pd.DataFrame): DataFrame with scraped content
        """
        text_dir = os.path.join(self.output_dir, 'text_files')
        if not os.path.exists(text_dir):
            os.makedirs(text_dir)
        
        for i, row in df.iterrows():
            # Create a filename based on title or index
            if row['title']:
                # Clean the title for use as a filename
                filename = re.sub(r'[^\w\s-]', '', row['title'])
                filename = re.sub(r'\s+', '_', filename)
                filename = filename[:50]  # Limit length
            else:
                filename = f"document_{i}"
                
            filepath = os.path.join(text_dir, f"{filename}.txt")
            
            # Write content to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Title: {row['title']}\n")
                f.write(f"URL: {row['url']}\n")
                f.write(f"Scraped at: {row['scraped_at']}\n\n")
                f.write(row['content'])
                
        print(f"Text files saved to {text_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Web scraper for creating text datasets')
    parser.add_argument('--urls', nargs='+', required=True, help='URLs to scrape')
    parser.add_argument('--output', default='dataset', help='Output directory')
    parser.add_argument('--max-pages', type=int, default=50, help='Max pages to scrape')
    
    args = parser.parse_args()
    
    scraper = WebScraper(args.urls, args.output)
    df = scraper.create_dataset(max_pages=args.max_pages)
    scraper.save_text_files(df)