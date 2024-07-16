# Sentiment Analysis with BERT

This project demonstrates how to fine-tune a BERT model for sentiment analysis using PyTorch and Huggingface Transformers and Datasets libraries. The IMDB dataset is used for training and evaluating the model.

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Overview

The goal of this project is to fine-tune a pretrained BERT model for sentiment analysis. The IMDB dataset, which contains movie reviews labeled as positive or negative, is used for this task. The project involves:

1. Loading and preprocessing the IMDB dataset.
2. Tokenizing the text data using BERT tokenizer.
3. Freezing the BERT model parameters.
4. Adding a custom classification head.
5. Training the classification head while keeping the BERT model parameters frozen.
6. Saving the model with the best performance.
7. Loading a pretrained model if available.

## Setup

### Requirements

- Python 3.6+
- PyTorch
- Huggingface Transformers
- Datasets
- scikit-learn

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/sentiment-analysis-bert.git
    cd sentiment-analysis-bert
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Requirements.txt
```txt
torch
transformers
datasets
scikit-learn
