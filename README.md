
# ArXiv Data Classification

This project aims to classify research papers from the ArXiv repository based on their abstracts. The classification is performed using machine learning techniques, specifically Natural Language Processing (NLP) and deep learning models.

### Project Overview

The main objectives of this project are:

1. **Data Collection**: Retrieve and preprocess data from the ArXiv repository, including paper abstracts, titles, and subject categories.
2. **Text Preprocessing**: Clean and prepare the text data for further processing, including tokenization, stopword removal, and stemming/lemmatization.
3. **Model Training**: Train and evaluate various machine learning models, including traditional classifiers (e.g., Naive Bayes, Logistic Regression) and deep learning models (e.g., Transformer Models such as Bert and SciBERT) for text classification.
4. **Model Evaluation**: Assess the performance of the trained models using appropriate evaluation metrics, such as accuracy, precision, recall, and F1-score.

Following this, the main research question was to explore whether the text length of the abstract has an effect on the classification performance and how this would differ across different models. As such, each model is subjected to truncated abstracts, assessing the performance of the classification task at incremental text lengths
