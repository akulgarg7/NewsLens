import os
import re
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Securely download necessary NLTK data (avoid blocking or repeated downloads)
def init_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)

# Run initialization
init_nltk()

def get_openai_client():
    """
    Initializes and returns the OpenAI client configured for Nararouter.
    """
    api_key = os.environ.get("NARAROUTER_API_KEY")
    base_url = os.environ.get("NARAROUTER_BASE_URL", "https://router.bynara.id/v1")
    
    if not api_key:
        raise ValueError("NARAROUTER_API_KEY is not set in environment or .env file.")
        
    return OpenAI(api_key=api_key, base_url=base_url)

def fetch_article_text(url):
    """
    Fetches the main body text of an article from a URL.
    Uses basic BeautifulSoup logic to strip headers, footers, navbars, and extract paragraphs.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        raise Exception(f"Failed to fetch the URL: {str(e)}")
        
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Remove script, style, header, footer, nav, and ads elements
    for element in soup(["script", "style", "header", "footer", "nav", "aside", "form"]):
        element.decompose()
        
    # Standard news websites put text inside article tags or generic paragraph elements
    paragraphs = soup.find_all('p')
    
    article_text = "\n".join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30])
    
    if not article_text:
        # Fallback to general body text if no paragraphs met length requirements
        article_text = soup.get_text(separator='\n').strip()
        
    # Clean multiple consecutive newlines or whitespace
    article_text = re.sub(r'\n+', '\n', article_text)
    article_text = re.sub(r' {2,}', ' ', article_text)
    
    return article_text

def clean_and_normalize_text(text):
    """
    Cleans raw text by removing punctuation, digits, extra whitespaces, etc.
    Returns lowered text containing alphanumeric characters and basic punctuation.
    """
    # Lowercase
    cleaned = text.lower()
    # Normalize spacing
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def perform_nlp_preprocessing(text):
    """
    Performs sentence tokenization, word tokenization, and stopword removal.
    Returns:
    - total_words: count of raw words
    - total_sentences: count of sentences
    - word_frequencies: Counter object of top words (excluding stopwords and punctuation)
    - cleaned_tokens: list of cleaned non-stopword tokens
    """
    init_nltk()
    sentences = nltk.sent_tokenize(text)
    raw_tokens = nltk.word_tokenize(text)
    
    # Filter stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = []
    
    for token in raw_tokens:
        token_lower = token.lower()
        # Ensure it contains at least one alphabetic character and isn't a stopword or short punctuation
        if token_lower.isalpha() and token_lower not in stop_words and len(token_lower) > 1:
            cleaned_tokens.append(token_lower)
            
    word_frequencies = Counter(cleaned_tokens)
    
    return {
        "total_words": len(raw_tokens),
        "total_sentences": len(sentences),
        "cleaned_tokens": cleaned_tokens,
        "word_frequencies": word_frequencies
    }

def get_vader_sentiment(text):
    """
    Calculates Sentiment using NLTK's classical VADER lexicon.
    Returns a dictionary of scores (pos, neu, neg, compound) and a label.
    """
    init_nltk()
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    
    compound = scores['compound']
    if compound >= 0.05:
        label = "Positive"
        color = "#10B981"  # Emerald Green
    elif compound <= -0.05:
        label = "Negative"
        color = "#EF4444"  # Coral Red
    else:
        label = "Neutral"
        color = "#6B7280"  # Gray
        
    return {
        "scores": scores,
        "label": label,
        "color": color
    }

def get_sentiment_progression(text):
    """
    Computes sentiment progression across an article by sentence-tokenizing,
    calculating compound scores, and applying a sliding window average.
    """
    init_nltk()
    sentences = nltk.sent_tokenize(text)
    sia = SentimentIntensityAnalyzer()
    
    results = []
    sentence_scores = []
    
    # Calculate raw scores per sentence
    for i, sent in enumerate(sentences):
        sent_clean = sent.strip()
        if not sent_clean:
            continue
        score = sia.polarity_scores(sent_clean)["compound"]
        sentence_scores.append((i + 1, score, sent_clean[:60] + "..."))
        
    if not sentence_scores:
        return []
        
    # Apply rolling window of 3 sentences to smooth the emotional trend
    window = 3
    for idx, (sent_num, raw_score, snippet) in enumerate(sentence_scores):
        start_idx = max(0, idx - window + 1)
        sub_scores = [item[1] for item in sentence_scores[start_idx:idx + 1]]
        smoothed_score = sum(sub_scores) / len(sub_scores)
        results.append({
            "Sentence Index": sent_num,
            "Raw Sentiment": raw_score,
            "Rolling Sentiment": round(smoothed_score, 4),
            "Snippet": snippet
        })
        
    return results

def query_nararouter(prompt, system_prompt="You are a helpful assistant."):
    """
    Queries the Nararouter API using the custom credentials and model name.
    """
    try:
        client = get_openai_client()
        model_name = os.environ.get("NARAROUTER_MODEL_NAME", "mistral-medium-3-5")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API Query Error: {str(e)}"

import json
from datetime import datetime

def log_activity(activity_type, details):
    """
    Logs user activity to a local JSON file (activity_history.json) for auditing and persistence.
    Designed to be robust and fail-safe, working seamlessly on local systems as well as Streamlit Cloud.
    """
    filepath = "activity_history.json"
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "activity_type": activity_type,
        "details": details
    }
    
    # Read existing history
    history = []
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = []
        except Exception:
            history = []
            
    # Append and write back
    history.append(new_entry)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing to activity_history.json: {str(e)}")

def get_activity_history():
    """
    Retrieves the activity history list from activity_history.json.
    """
    filepath = "activity_history.json"
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                history = json.load(f)
                if isinstance(history, list):
                    return history
        except Exception:
            pass
    return []

def clear_activity_history():
    """
    Clears the activity history by overwriting with an empty JSON list.
    """
    filepath = "activity_history.json"
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([], f)
    except Exception:
        pass
