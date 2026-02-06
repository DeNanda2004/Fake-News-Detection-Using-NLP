import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import preprocessing
import os

# Define paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/news.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model.pkl')

def train_models():
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    print("Loading data...")
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"Error loading csv: {e}")
        return

    print(f"Data shape: {df.shape}")
    
    # Check necessary columns
    if 'text' not in df.columns or 'label' not in df.columns:
        print("Error: CSV must contain 'text' and 'label' columns.")
        return

    # 2. Preprocess
    print("Preprocessing text...")
    df['processed_text'] = df['text'].apply(lambda x: preprocessing.preprocess_text(x, method='stemming'))
    
    # 3. Split Data
    X = df['processed_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Define Models
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(probability=True)
    }
    
    best_model = None
    best_accuracy = -1 # Start with -1 so even 0 overrides it
    best_model_name = ""
    
    results = {}

    for name, clf in models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', clf)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = pipeline
            best_model_name = name

    print("-" * 30)
    print("Model Comparison:")
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")
        
    print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
    
    # 5. Save Best Model
    print(f"Saving best model to {MODEL_PATH}...")
    joblib.dump(best_model, MODEL_PATH)
    print("Done.")

if __name__ == "__main__":
    train_models()
