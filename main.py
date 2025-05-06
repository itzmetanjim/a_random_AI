#!/usr/bin/env python
# Main script to run the QA system with web scraping
import argparse
import os
import pandas as pd
from web_scraper import WebScraper
from llm_qa_system import QASystem

def main():
    parser = argparse.ArgumentParser(description='Web Scraping and QA System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Scraper command
    scraper_parser = subparsers.add_parser('scrape', help='Scrape websites to create dataset')
    scraper_parser.add_argument('--urls', nargs='+', required=True, help='URLs to scrape')
    scraper_parser.add_argument('--output', default='dataset', help='Output directory')
    scraper_parser.add_argument('--max-pages', type=int, default=50, help='Max pages to scrape')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the QA model')
    train_parser.add_argument('--dataset', required=True, help='Path to dataset CSV')
    train_parser.add_argument('--model-dir', default='qa_model', help='Model directory')
    train_parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    
    # QA command
    qa_parser = subparsers.add_parser('qa', help='Answer questions')
    qa_parser.add_argument('--model-dir', default='qa_model', help='Model directory')
    qa_parser.add_argument('--dataset', required=True, help='Path to dataset CSV for context')
    qa_parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    qa_parser.add_argument('--question', help='Question to answer (non-interactive mode)')
    
    args = parser.parse_args()
    
    if args.command == 'scrape':
        print(f"Starting web scraping with URLs: {args.urls}")
        scraper = WebScraper(args.urls, args.output)
        df = scraper.create_dataset(max_pages=args.max_pages)
        scraper.save_text_files(df)
        print(f"Scraping completed. Dataset saved to {os.path.join(args.output, 'scraped_data.csv')}")
        
    elif args.command == 'train':
        print(f"Training QA model with dataset: {args.dataset}")
        qa_system = QASystem(args.model_dir)
        
        # Load dataset
        dataset_df = pd.read_csv(args.dataset)
        
        # Create QA pairs
        qa_df = qa_system.create_qa_pairs(dataset_df, 'content', 'title', 'url')
        
        # Train model
        print(f"Generated {len(qa_df)} QA pairs for training")
        qa_system.train_model(qa_df, epochs=args.epochs)
        print(f"Model trained and saved to {args.model_dir}")
        
    elif args.command == 'qa':
        print(f"Loading QA model from {args.model_dir}")
        qa_system = QASystem(args.model_dir)
        
        # Load model
        if not qa_system.load_model():
            print("Failed to load model. Please train the model first.")
            return
            
        # Load dataset for context
        dataset_df = pd.read_csv(args.dataset)
        
        if args.interactive:
            # Interactive QA mode
            print("Starting interactive QA mode. Type 'exit' to quit.")
            while True:
                question = input("\nEnter your question: ")
                if question.lower() == "exit":
                    break
                    
                answer = qa_system.answer_question(question, dataset_df=dataset_df)
                print(f"Answer: {answer}")
                print("-" * 50)
        else:
            # Single question mode
            if not args.question:
                print("Please provide a question with --question or use --interactive mode")
                return
                
            answer = qa_system.answer_question(args.question, dataset_df=dataset_df)
            print(f"Question: {args.question}")
            print(f"Answer: {answer}")
    
    else:
        parser.print_help()
        
if __name__ == "__main__":
    main()