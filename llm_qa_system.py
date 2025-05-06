#!/usr/bin/env python
# Question Answering System using TensorFlow
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Model
import pickle
import re
import argparse
from sklearn.model_selection import train_test_split

class QASystem:
    def __init__(self, model_dir="qa_model"):
        """
        Initialize the QA system
        
        Args:
            model_dir (str): Directory to save/load the model
        """
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        self.max_sequence_length = 200
        self.embedding_dim = 300
        self.vocab_size = 10000
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
    def preprocess_text(self, text):
        """
        Preprocess text by cleaning and normalizing
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra spaces
        text = re.sub(r'[^\w\s.,?!]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def create_qa_pairs(self, df, text_column, title_column=None, url_column=None):
        """
        Create question-answer pairs from text data
        
        Args:
            df (pd.DataFrame): DataFrame with text data
            text_column (str): Column containing the document text
            title_column (str): Column containing the document title
            url_column (str): Column containing the document URL
            
        Returns:
            pd.DataFrame: DataFrame with question-answer pairs
        """
        qa_pairs = []
        
        for _, row in df.iterrows():
            text = row[text_column]
            
            # Skip empty texts
            if not text:
                continue
                
            # Clean and split text into sentences
            text = self.preprocess_text(text)
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Get metadata
            title = row.get(title_column, "") if title_column else ""
            url = row.get(url_column, "") if url_column else ""
            
            # Generate QA pairs from text
            for i, sentence in enumerate(sentences):
                if len(sentence.split()) < 5:  # Skip very short sentences
                    continue
                    
                # Context is surrounding sentences
                start_idx = max(0, i-2)
                end_idx = min(len(sentences), i+3)
                context = " ".join(sentences[start_idx:end_idx])
                
                # For training examples, use the sentence as the answer
                # and create questions based on the sentence
                answer = sentence
                
                # Create different question types
                # 1. What/who question
                if any(word in sentence.lower() for word in ["is", "are", "was", "were"]):
                    question = f"What {sentence.split()[1:3]}"
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'context': context,
                        'title': title,
                        'url': url
                    })
                
                # 2. Question based on entities in the sentence
                words = sentence.split()
                if len(words) > 5:
                    entity = " ".join(words[:2])
                    question = f"What about {entity}?"
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'context': context,
                        'title': title,
                        'url': url
                    })
                
                # 3. Generic question about the content
                question = f"What information is provided about {title if title else 'this topic'}?"
                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'context': context,
                    'title': title,
                    'url': url
                })
        
        return pd.DataFrame(qa_pairs)
    
    def prepare_training_data(self, qa_df):
        """
        Prepare training data for the model
        
        Args:
            qa_df (pd.DataFrame): DataFrame with question-answer pairs
            
        Returns:
            tuple: Training data for the model
        """
        # Combine question and context for input
        inputs = []
        outputs = []
        
        for _, row in qa_df.iterrows():
            question = row['question']
            context = row['context']
            answer = row['answer']
            
            # Create input as "question [SEP] context"
            input_text = f"{question} [SEP] {context}"
            inputs.append(input_text)
            outputs.append(answer)
        
        # Create tokenizer
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(inputs + outputs)
        
        # Convert texts to sequences
        input_sequences = self.tokenizer.texts_to_sequences(inputs)
        output_sequences = self.tokenizer.texts_to_sequences(outputs)
        
        # Pad sequences
        padded_inputs = pad_sequences(input_sequences, maxlen=self.max_sequence_length, padding='post')
        padded_outputs = pad_sequences(output_sequences, maxlen=self.max_sequence_length, padding='post')
        
        # Save tokenizer
        with open(os.path.join(self.model_dir, 'tokenizer.pickle'), 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return padded_inputs, padded_outputs
    
    def build_model(self):
        """
        Build the QA model architecture
        
        Returns:
            Model: Compiled TensorFlow model
        """
        # Input layer
        input_layer = Input(shape=(self.max_sequence_length,))
        
        # Embedding layer
        embedding_layer = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_sequence_length
        )(input_layer)
        
        # Bidirectional LSTM layers
        lstm_layer1 = Bidirectional(LSTM(256, return_sequences=True))(embedding_layer)
        dropout1 = Dropout(0.2)(lstm_layer1)
        
        lstm_layer2 = Bidirectional(LSTM(128))(dropout1)
        dropout2 = Dropout(0.2)(lstm_layer2)
        
        # Dense layers
        dense1 = Dense(256, activation='relu')(dropout2)
        dropout3 = Dropout(0.2)(dense1)
        
        # Output layer
        output_layer = Dense(self.vocab_size, activation='softmax')(dropout3)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output_layer)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, qa_df, epochs=10, batch_size=32, validation_split=0.2):
        """
        Train the QA model
        
        Args:
            qa_df (pd.DataFrame): DataFrame with question-answer pairs
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data to use for validation
            
        Returns:
            Model: Trained model
        """
        # Prepare training data
        inputs, outputs = self.prepare_training_data(qa_df)
        
        # Build model
        self.model = self.build_model()
        
        # Train model
        history = self.model.fit(
            inputs,
            outputs,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )
        
        # Save model
        self.model.save(os.path.join(self.model_dir, 'qa_model'))
        
        return self.model, history
    
    def load_model(self):
        """
        Load a pretrained model and tokenizer
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Load tokenizer
            with open(os.path.join(self.model_dir, 'tokenizer.pickle'), 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            
            # Load model
            model_path = os.path.join(self.model_dir, 'qa_model')
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                return True
            else:
                print(f"Model not found at {model_path}")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
            
    def answer_question(self, question, context=None, dataset_df=None):
        """
        Answer a question using the trained model
        
        Args:
            question (str): Question to answer
            context (str): Optional context for the question
            dataset_df (pd.DataFrame): Optional dataset to search for relevant context
            
        Returns:
            str: Answer to the question
        """
        if not self.model or not self.tokenizer:
            if not self.load_model():
                return "Model not loaded. Please train the model first."
        
        # If no context provided, try to find relevant context from dataset
        if not context and dataset_df is not None:
            # Simple search for now - find most relevant document
            question_tokens = set(self.preprocess_text(question).split())
            
            max_overlap = 0
            best_context = ""
            
            for _, row in dataset_df.iterrows():
                content = self.preprocess_text(row['content'])
                content_tokens = set(content.split())
                overlap = len(question_tokens.intersection(content_tokens))
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    # Use a snippet of the content as context
                    words = content.split()
                    if len(words) > 200:
                        # Find segment with most question words
                        best_segment_overlap = 0
                        best_segment = ""
                        
                        for i in range(0, len(words) - 100, 50):
                            segment = " ".join(words[i:i+100])
                            segment_tokens = set(segment.split())
                            segment_overlap = len(question_tokens.intersection(segment_tokens))
                            
                            if segment_overlap > best_segment_overlap:
                                best_segment_overlap = segment_overlap
                                best_segment = segment
                                
                        best_context = best_segment
                    else:
                        best_context = content
            
            context = best_context
        
        # Use empty context if still None
        if not context:
            context = ""
            
        # Preprocess input
        input_text = f"{self.preprocess_text(question)} [SEP] {self.preprocess_text(context)}"
        input_sequence = self.tokenizer.texts_to_sequences([input_text])
        padded_input = pad_sequences(input_sequence, maxlen=self.max_sequence_length, padding='post')
        
        # Get prediction
        prediction = self.model.predict(padded_input)[0]
        
        # Convert prediction to text
        # Get top predicted token indices
        top_indices = np.argsort(prediction)[-20:]  # Take top 20 tokens
        
        # Convert indices to words
        index_to_word = {v: k for k, v in self.tokenizer.word_index.items()}
        answer_words = [index_to_word.get(i, "") for i in top_indices if i > 0]
        
        # Filter out empty words and join
        answer = " ".join([w for w in answer_words if w])
        
        return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Question Answering System')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--dataset', required=True, help='Path to dataset CSV')
    parser.add_argument('--model-dir', default='qa_model', help='Model directory')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    
    args = parser.parse_args()
    
    qa_system = QASystem(args.model_dir)
    
    if args.train:
        # Load dataset
        print(f"Loading dataset from {args.dataset}")
        dataset_df = pd.read_csv(args.dataset)
        
        # Create QA pairs
        qa_df = qa_system.create_qa_pairs(dataset_df, 'content', 'title', 'url')
        
        # Train model
        print(f"Training model with {len(qa_df)} QA pairs")
        qa_system.train_model(qa_df, epochs=args.epochs)
        print(f"Model trained and saved to {args.model_dir}")
    else:
        # Interactive QA mode
        if qa_system.load_model():
            print("Model loaded successfully")
            
            # Load dataset for context
            dataset_df = pd.read_csv(args.dataset)
            
            while True:
                question = input("Enter your question (or 'exit' to quit): ")
                if question.lower() == "exit":
                    break
                    
                answer = qa_system.answer_question(question, dataset_df=dataset_df)
                print(f"Answer: {answer}")
                print("-" * 50)
        else:
            print("Failed to load model. Please train the model first.")