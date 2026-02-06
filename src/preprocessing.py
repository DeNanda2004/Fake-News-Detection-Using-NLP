import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Ensure necessary NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt_tab')

def clean_text(text):
    """
    Cleans the input text by:
    1. Lowercasing
    2. Removing URLS
    3. Removing punctuation and numbers
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'\[.*?\]', '', text) 
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # Remove punctuation
    text = re.sub(r'\w*\d\w*', '', text) # Remove words containing numbers
    text = re.sub(r'\n', ' ', text)
    return text

def preprocess_text(text, method='stemming'):
    """
    Full preprocessing pipeline:
    1. Clean
    2. Tokenize
    3. Remove stopwords
    4. Stemming or Lemmatization
    """
    text = clean_text(text)
    
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    
    if method == 'stemming':
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(w) for w in tokens]
    elif method == 'lemmatization':
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
        
    return " ".join(tokens)

if __name__ == "__main__":
    # Test
    sample = "Breaking News: Aliens landed in New York! Visit http://fake.com 123"
    print(f"Original: {sample}")
    print(f"Processed: {preprocess_text(sample)}")
