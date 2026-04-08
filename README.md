# Sentiment Analysis Tool (CLI-based)

This project implements a **Sentiment Analysis Tool** using machine learning to classify text into **positive, negative, or neutral sentiments**.



## Features

* Text preprocessing (cleaning and normalization)
* Feature extraction using TF-IDF Vectorizer
* Classification using Naive Bayes algorithm
* Supports positive, negative, and neutral sentiments
* Interactive command-line interface (CLI)
* Model evaluation using accuracy score


## How It Works

1. Input text is cleaned (lowercased, punctuation removed)
2. Text is converted into numerical features using TF-IDF
3. A trained Naive Bayes model predicts sentiment
4. Output is displayed instantly


##  Project Structure

* `sentiment.py` → Main script (training + prediction)



## How to Run

### 1. Install Dependencies

```bash
pip install scikit-learn
```

### 2. Run the Program

```bash
python sentiment.py
```


##  Example Usage

Input:
I love this product

Output:
Predicted Sentiment: positive


##  Model Details

* Algorithm: Multinomial Naive Bayes
* Feature Extraction: TF-IDF
* Dataset: Custom labeled dataset (~30+ samples)
* Evaluation: Accuracy score



##  Technologies Used

* Python
* Scikit-learn
* Regular Expressions


##  Highlights

* Demonstrates basic Natural Language Processing (NLP)
* Implements complete ML pipeline
* Interactive and user-friendly
* Easily extendable with larger datasets


## Future Improvements

* Use real-world datasets (tweets/reviews)
* Apply advanced preprocessing techniques
* Experiment with other ML models
* Build GUI or web-based interface


##  Author

Developed as part of an AI Internship task, focusing on building a simple sentiment classification system.
