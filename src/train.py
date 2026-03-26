import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords

# Download stopwords if not available
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Clean the text by lowercasing, removing punctuation, and stop words."""
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def compute_priority(text):
    """
    Synthesize priority based on urgency keywords in the original text.
    In real scenarios, priority could be set by agents or SLAs. 
    Here, we simulate priority for the sake of the ML task.
    """
    text = str(text).lower()
    high_keywords = ['urgent', 'asap', 'critical', 'emergency', 'down', 'crash', 'immediate', 'broken', 'fail', 'ceo', 'meeting', 'minute', 'cannot access', 'locked out']
    low_keywords = ['info', 'query', 'question', 'how to', 'update', 'help with', 'request']
    
    if any(kw in text for kw in high_keywords):
        return 'High'
    elif any(kw in text for kw in low_keywords):
        return 'Low'
    else:
        return 'Medium'

def main():
    print("Loading dataset...")
    df = pd.read_csv('../data/all_tickets_processed_improved_v3.csv')
    
    # We will sample 10,000 or the max rows to speed up training if it's very large, 
    # but 14MB should be fast enough. Let's use the whole dataset.
    print(f"Dataset shape: {df.shape}")
    
    df = df.dropna(subset=['Document', 'Topic_group'])
    
    print("Assigning simulated priority levels based on text keywords...")
    df['Priority'] = df['Document'].apply(compute_priority)
    
    print("Cleaning text data...")
    df['Cleaned_Document'] = df['Document'].apply(clean_text)
    
    print("Splitting data into train and test sets...")
    X = df['Cleaned_Document']
    y_cat = df['Topic_group']
    y_prio = df['Priority']
    
    X_train, X_test, y_cat_train, y_cat_test, y_prio_train, y_prio_test = train_test_split(
        X, y_cat, y_prio, test_size=0.2, random_state=42
    )
    
    print("Extracting features with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("Training Category Classification Model...")
    # Logistic Regression is fast and generally performs well on text classification
    cat_model = LogisticRegression(max_iter=1000, solver='liblinear')
    cat_model.fit(X_train_vec, y_cat_train)
    
    print("Evaluating Category Model...")
    cat_preds = cat_model.predict(X_test_vec)
    print("Accuracy (Category):", accuracy_score(y_cat_test, cat_preds))
    print(classification_report(y_cat_test, cat_preds))
    
    print("Training Priority Prediction Model...")
    prio_model = LogisticRegression(max_iter=1000, solver='liblinear')
    prio_model.fit(X_train_vec, y_prio_train)
    
    print("Evaluating Priority Model...")
    prio_preds = prio_model.predict(X_test_vec)
    print("Accuracy (Priority):", accuracy_score(y_prio_test, prio_preds))
    print(classification_report(y_prio_test, prio_preds))
    
    print("Saving models to models.pkl...")
    joblib.dump({
        'vectorizer': vectorizer,
        'category_model': cat_model,
        'priority_model': prio_model
    }, '../models.pkl')
    print("Done! Models saved to models.pkl")

if __name__ == '__main__':
    main()
