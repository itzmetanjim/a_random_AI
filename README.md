# Web Scraping and Question Answering System

This project combines web scraping capabilities with a machine learning-based Question Answering system. It allows you to:

1. Scrape websites to build a custom dataset
2. Train a QA model on the dataset
3. Ask questions and get answers based on the scraped content

## System Components

- **Web Scraper**: Collects text data from websites
- **QA System**: TensorFlow-based neural network for answering questions
- **Main Script**: Ties everything together with a simple CLI

## Setup Instructions

### Prerequisites

Ensure you have Python 3.8+ installed, along with pip. Then install the required packages:

```bash
pip install requests beautifulsoup4 pandas numpy tensorflow scikit-learn
```

### Getting Started

1. **Clone or download this repository**

2. **Scrape a website to create your dataset:**

```bash
python main.py scrape --urls https://example.com --output my_dataset --max-pages 20
```

This will:
- Scrape up to 20 pages from example.com
- Save the data to my_dataset/scraped_data.csv
- Create individual text files in my_dataset/text_files/

3. **Train the QA model:**

```bash
python main.py train --dataset my_dataset/scraped_data.csv --model-dir my_qa_model --epochs 15
```

This will:
- Load the scraped data
- Create question-answer pairs automatically
- Train a TensorFlow model for 15 epochs
- Save the model to my_qa_model directory

4. **Use the QA system:**

Interactive mode:
```bash
python main.py qa --model-dir my_qa_model --dataset my_dataset/scraped_data.csv --interactive
```

Single question mode:
```bash
python main.py qa --model-dir my_qa_model --dataset my_dataset/scraped_data.csv --question "What is the main topic of this website?"
```

## Advanced Usage

### Custom Scraping Configuration

You can customize the web scraper by modifying `web_scraper.py`:

- Change content tags to target specific HTML elements
- Adjust scraping delays to be more respectful to websites
- Modify text cleaning functions for your specific needs

Example:
```bash
# Scrape multiple websites
python main.py scrape --urls https://site1.com https://site2.com --max-pages 50
```

### Training Options

The QA system can be tuned by:

- Increasing training epochs for better accuracy
- Adjusting the model architecture in `llm_qa_system.py`
- Adding more sophisticated question-answer pair generation

### Alternative Transformer-Based QA

For more advanced question answering capabilities, use the included transformer-based implementation:

```bash
python transformer_qa.py --dataset my_dataset/scraped_data.csv --question "What are the key points discussed in this website?"
```

## How It Works

1. **Web Scraper**:
   - Crawls websites starting from provided URLs
   - Extracts text content from various HTML elements
   - Cleans and processes text data
   - Saves structured data as CSV and text files

2. **QA System**:
   - Uses neural networks to understand questions and context
   - Automatically generates training examples from scraped content
   - Leverages bidirectional LSTM architecture for text understanding
   - Provides relevant answers based on the context

3. **Transformer QA** (Alternative):
   - Uses pre-trained transformer models from HuggingFace
   - Provides state-of-the-art question answering capabilities
   - Can handle complex questions with better natural language understanding

## Limitations

- The basic QA model may need significant training data for good results
- Scraped data quality depends on website structure
- Basic model might struggle with complex questions

## Future Improvements

- Add more sophisticated text cleaning and preprocessing
- Implement retrieval-augmented generation for better answers
- Add support for PDF and other document formats
- Create a web interface for easier interaction

## License

This project is open source and available under the MIT License.