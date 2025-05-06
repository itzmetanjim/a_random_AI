#!/usr/bin/env python
# Advanced Question Answering System using HuggingFace Transformers
import os
import re
import pandas as pd
import numpy as np
import argparse
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    pipeline,
    BertTokenizer,
    BertForQuestionAnswering
)
from sklearn.feature_extraction.text import TfidfVectorizer
from web_scraper import WebScraper

class TransformerQA:
    def __init__(self, model_name=None, use_seq2seq=False):
        """
        Initialize the transformer-based QA system
        
        Args:
            model_name (str): HuggingFace model name to use
            use_seq2seq (bool): Whether to use a seq2seq model (T5, BART) instead of QA model
        """
        self.use_seq2seq = use_seq2seq
        
        # Default to BERT for extractive QA if not specified
        if model_name is None:
            self.model_name = "bert-large-uncased-whole-word-masking-finetuned-squad" if not use_seq2seq else "t5-base"
            
        # Choose appropriate model type based on flag
        if use_seq2seq:
            print(f"Loading seq2seq model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.nlp = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)
        else:
            print(f"Loading QA model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
            self.nlp = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer)
            
        # For context retrieval
        self.vectorizer = None
        self.document_vectors = None
        self.documents = None
            
    def preprocess_text(self, text):
        """
        Clean and preprocess text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
            
        # Remove special characters, URLs, and extra whitespace
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s.,?!\(\)\[\]"\':-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    def index_documents(self, documents):
        """
        Index documents for retrieval
        
        Args:
            documents (list): List of document texts
            
        Returns:
            self: For method chaining
        """
        print(f"Indexing {len(documents)} documents...")
        self.documents = [self.preprocess_text(doc) for doc in documents]
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=2
        )
        
        # Fit and transform documents
        self.document_vectors = self.vectorizer.fit_transform(self.documents)
        
        return self
        
    def get_relevant_context(self, question, top_k=3, max_len=3000):
        """
        Retrieve most relevant document segments for a question using improved
        context retrieval with sentence-level chunking and keyword matching
        
        Args:
            question (str): Question to answer
            top_k (int): Number of top segments to retrieve
            max_len (int): Maximum context length
            
        Returns:
            str: Concatenated context text with the most relevant information
        """
        if self.vectorizer is None or self.document_vectors is None:
            return ""
        
        # Extract keywords from the question
        # Remove common question words and stop words
        question_clean = re.sub(r'\b(what|who|where|when|why|how|is|are|do|does|did|can|could|would|should)\b', 
                                '', question.lower())
        question_keywords = set(re.findall(r'\b\w+\b', question_clean))
        question_keywords = {w for w in question_keywords if len(w) > 2}
        
        # Vectorize question
        question_vector = self.vectorizer.transform([question])
        
        # Calculate similarity
        similarities = (self.document_vectors @ question_vector.T).toarray().flatten()
        
        # Get top k*2 most similar documents (we'll filter further)
        top_indices = similarities.argsort()[-(top_k*2):][::-1]
        
        # Segment documents into more manageable chunks (sentences or paragraphs)
        chunks = []
        chunk_doc_indices = []  # Keep track of which document each chunk belongs to
        
        for doc_idx in top_indices:
            doc = self.documents[doc_idx]
            
            # Split into sentences or smaller chunks
            sentences = re.split(r'(?<=[.!?])\s+', doc)
            
            # Group sentences into slightly larger chunks for better context
            for i in range(0, len(sentences), 3):
                chunk_text = " ".join(sentences[i:i+5])  # Overlap for context
                if chunk_text.strip():
                    chunks.append(chunk_text)
                    chunk_doc_indices.append(doc_idx)
        
        # Score chunks based on keyword match and original document similarity
        chunk_scores = []
        for i, chunk in enumerate(chunks):
            # Base score from document similarity
            doc_idx = chunk_doc_indices[i]
            base_score = similarities[doc_idx]
            
            # Keyword matching score
            chunk_lower = chunk.lower()
            keyword_matches = sum(1 for kw in question_keywords if kw in chunk_lower)
            keyword_score = keyword_matches / max(1, len(question_keywords))
            
            # Position bonus (sentences at beginning of documents often contain key info)
            position_in_doc = chunks.index(chunk) / max(1, len(chunks))
            position_score = max(0, 1 - position_in_doc)  # Higher score for earlier chunks
            
            # Combined score with appropriate weights
            total_score = (base_score * 0.4) + (keyword_score * 0.5) + (position_score * 0.1)
            chunk_scores.append(total_score)
        
        # Sort chunks by score
        sorted_chunks = [x for _, x in sorted(zip(chunk_scores, chunks), reverse=True)]
        
        # Build context from top chunks
        context = ""
        for chunk in sorted_chunks:
            if len(context) + len(chunk) + 1 <= max_len:
                context += chunk + " "
            else:
                break
        
        # If we still have room, add introductory paragraphs from top documents
        if len(context) < (max_len * 0.7):
            for doc_idx in top_indices:
                doc = self.documents[doc_idx]
                paragraphs = re.split(r'\n+', doc)
                if paragraphs and len(paragraphs[0]) > 50:  # Only add substantial first paragraphs
                    if len(context) + len(paragraphs[0]) + 1 <= max_len:
                        # Check if this paragraph is already included
                        if paragraphs[0] not in context:
                            context = paragraphs[0] + " " + context  # Prepend for context
                
                if len(context) >= (max_len * 0.9):
                    break
                
        return context.strip()
        
    def segment_long_document(self, document, max_len=1000, overlap=200):
        """
        Split long documents into overlapping segments
        
        Args:
            document (str): Document text
            max_len (int): Maximum segment length
            overlap (int): Overlap between segments
            
        Returns:
            list: List of document segments
        """
        words = document.split()
        segments = []
        
        if len(words) <= max_len:
            return [document]
            
        for i in range(0, len(words), max_len - overlap):
            segment = " ".join(words[i:i + max_len])
            segments.append(segment)
            
            if i + max_len >= len(words):
                break
                
        return segments
        
    def answer_question(self, question, context=None, dataset_df=None, confidence_threshold=0.1):
        """
        Answer a question using transformer models
        
        Args:
            question (str): Question to answer
            context (str): Optional context for the question
            dataset_df (pd.DataFrame): Optional dataset to search for context
            confidence_threshold (float): Threshold for answer confidence
            
        Returns:
            dict: Answer information
        """
        # If no context provided but we have a dataset, find relevant context
        if not context and dataset_df is not None:
            # First, index documents if not already done
            if self.document_vectors is None:
                contents = dataset_df['content'].tolist()
                self.index_documents(contents)
                
            # Get relevant context
            context = self.get_relevant_context(question)
            
        # Fallback for no context
        if not context:
            context = "No relevant information found."
            
        # For very long contexts, we need to handle differently
        if len(context.split()) > 512:
            # For seq2seq models, we can just truncate
            if self.use_seq2seq:
                # Truncate to model's max input length (usually around 512 tokens)
                words = context.split()
                context = " ".join(words[:512])
                
                # Generate answer
                result = self.nlp(f"question: {question} context: {context}", 
                                   max_length=100, 
                                   num_beams=4)
                
                return {
                    "answer": result[0]["generated_text"],
                    "context": context[:100] + "..." if len(context) > 100 else context,
                    "score": 1.0  # Seq2seq models don't provide confidence scores in this way
                }
            else:
                # For extractive QA, segment the document and find best answer
                segments = self.segment_long_document(context)
                best_answer = None
                best_score = 0
                
                for segment in segments:
                    try:
                        result = self.nlp(question=question, context=segment)
                        
                        if result["score"] > best_score:
                            best_score = result["score"]
                            best_answer = result
                    except Exception as e:
                        print(f"Error processing segment: {e}")
                        continue
                
                if best_answer and best_score > confidence_threshold:
                    return best_answer
                else:
                    return {"answer": "I couldn't find a reliable answer in the provided context.", 
                            "score": 0.0,
                            "context": context[:100] + "..." if len(context) > 100 else context}
        else:
            # Standard approach for manageable context length
            try:
                if self.use_seq2seq:
                    result = self.nlp(f"question: {question} context: {context}", 
                                    max_length=100, 
                                    num_beams=4)
                    return {
                        "answer": result[0]["generated_text"],
                        "context": context[:100] + "..." if len(context) > 100 else context,
                        "score": 1.0
                    }
                else:
                    result = self.nlp(question=question, context=context)
                    return result
            except Exception as e:
                print(f"Error answering question: {e}")
                return {"answer": "An error occurred while processing the question.", 
                        "score": 0.0,
                        "context": context[:100] + "..." if len(context) > 100 else context}

    def load_dataset(self, csv_path):
        """
        Load dataset from CSV file
        
        Args:
            csv_path (str): Path to CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            df = pd.read_csv(csv_path)
            # Index documents for retrieval
            if 'content' in df.columns:
                contents = df['content'].tolist()
                self.index_documents(contents)
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description='Transformer-based Question Answering System')
    parser.add_argument('--model', default=None, help='HuggingFace model name')
    parser.add_argument('--dataset', required=True, help='Path to dataset CSV')
    parser.add_argument('--seq2seq', action='store_true', help='Use seq2seq model instead of QA model')
    parser.add_argument('--question', help='Question to answer (non-interactive mode)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--scrape', action='store_true', help='Scrape new data first')
    parser.add_argument('--urls', nargs='+', help='URLs to scrape if --scrape is used')
    parser.add_argument('--output', default='dataset', help='Output directory for scraped data')
    parser.add_argument('--max-pages', type=int, default=20, help='Max pages to scrape')
    
    args = parser.parse_args()
    
    # Handle scraping if requested
    if args.scrape:
        if not args.urls:
            print("Please provide URLs to scrape with --urls")
            return
            
        print(f"Scraping data from: {args.urls}")
        scraper = WebScraper(args.urls, args.output)
        dataset_df = scraper.create_dataset(max_pages=args.max_pages)
        scraper.save_text_files(dataset_df)
        print(f"Scraping completed. Dataset saved to {os.path.join(args.output, 'scraped_data.csv')}")
        
        # Use the newly created dataset
        dataset_path = os.path.join(args.output, 'scraped_data.csv')
    else:
        dataset_path = args.dataset
    
    # Initialize the QA system
    qa_system = TransformerQA(model_name=args.model, use_seq2seq=args.seq2seq)
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    dataset_df = qa_system.load_dataset(dataset_path)
    
    if dataset_df is None:
        print(f"Failed to load dataset from {dataset_path}")
        return
        
    # Interactive or single question mode
    if args.interactive:
        print("Starting interactive QA mode. Type 'exit' to quit.")
        while True:
            question = input("\nEnter your question: ")
            if question.lower() == "exit":
                break
                
            print("Finding answer...")
            result = qa_system.answer_question(question, dataset_df=dataset_df)
            
            print(f"\nQuestion: {question}")
            print(f"Answer: {result['answer']}")
            if 'score' in result:
                print(f"Confidence: {result['score']:.2f}")
            print("-" * 50)
    elif args.question:
        print(f"Question: {args.question}")
        print("Finding answer...")
        result = qa_system.answer_question(args.question, dataset_df=dataset_df)
        
        print(f"Answer: {result['answer']}")
        if 'score' in result:
            print(f"Confidence: {result['score']:.2f}")
    else:
        print("Please provide a question with --question or use --interactive mode")


if __name__ == "__main__":
    main()