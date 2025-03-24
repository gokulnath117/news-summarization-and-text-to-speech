import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from newspaper import Article
import numpy as np
from collections import Counter
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gtts import gTTS
from deep_translator import GoogleTranslator
import io
import yake
import json
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
gemini_api_key = os.getenv("Gemini_api")
genai.configure(api_key=gemini_api_key)


vader = SentimentIntensityAnalyzer()

def get_news_urls(company, max_results=20):
    """Fetches news article URLs related to a given company from Bing News"""
    search_url = f"https://www.google.com/search?q={company.replace(' ', '+')}+news&tbm=nws"

    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    remove_url=['https://www.google.com','https://maps.google.com','https://play.google.com','https://policies.google.com','https://support.google.com','https://accounts.google']

    news_links = set()
    for link in soup.find_all('a', href=True):
        url = link['href']
        filter_url = url.split("/url?q=")[-1].split("&")[0]
        if "http" in filter_url and 'msn' not in filter_url and not any(remove in filter_url for remove in remove_url):
            news_links.add(filter_url)
            if len(news_links) >= max_results:
                break
    
    return list(news_links)

def extract_keywords(news_article, top_n=3):
    """
    Extracts the top 3 most important keywords from news summaries using YAKE.
    - top_n: Number of keywords to extract per article.
    """
    keyword_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=10)

    if news_article:
        
        keywords = keyword_extractor.extract_keywords(news_article)
        
        top_keywords = [kw for kw, _ in keywords[:top_n]]

        return top_keywords
    return None

def analyze_sentiment(news_article):
    """
    Performs sentiment analysis on news summaries.
    Uses VADER for short texts and TextBlob for detailed polarity scoring.
    """
    if news_article:
        # summary = news.get("summary", "")
        
        # VADER Sentiment Score
        vader_score = vader.polarity_scores(news_article)["compound"]
        
        # TextBlob Sentiment Score
        blob_score = TextBlob(news_article).sentiment.polarity

        # Combined Score (weighted avg)
        final_score = (vader_score + blob_score) / 2

        # Determine sentiment label
        if final_score > 0.2:
            sentiment = "Positive"
        elif final_score < -0.2:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
    
        return sentiment
    
    return None

def extract_news(url):
    """Scrapes article title, summary, and text using newspaper3k."""
    try:
        article = Article(url)
        article.download()
        article.parse()

        source = url.split('/')[2]
        keywords = extract_keywords(article.meta_description if article.meta_description else article.text[:250])
        sentiment=analyze_sentiment(article.meta_description if article.meta_description else article.text[:250])

        return {
            "title": article.title,
            "summary": article.meta_description if article.meta_description else article.text[:250],
            "source": source,
            "keywords": keywords,
            "publish_date": str(article.publish_date) if article.publish_date else "No Date Available",
            "sentiment":sentiment,
            "url": url
        }
    except Exception as e:
        # print(f"Error extracting {url}: {e}")
        return None
    
def remove_duplicate_news(news_articles, similarity_threshold=0.8):
    """Removes news articles with similar content using cosine similarity"""
    if not news_articles:
        return []

    texts = [article["summary"] for article in news_articles]
    
    # Compute TF-IDF vectors and cosine similarity
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    unique_articles = []
    seen_indices = set()

    for i, article in enumerate(news_articles):
        if i in seen_indices:
            continue
        unique_articles.append(article)
        for j in range(i + 1, len(news_articles)):
            if similarity_matrix[i, j] > similarity_threshold:
                seen_indices.add(j)  # Mark duplicate article

    return unique_articles

def get_unique_news(company):
    """Fetch unique news articles about a company"""
    news_urls = get_news_urls(company)

    if news_urls:
        all_news = []
        for url in news_urls:
            news = extract_news(url)
            if news:
                all_news.append(news)
                if len(all_news) >= 20:  # Fetch more to account for duplicate removal
                    break

        # print(all_news)

        # Remove duplicate content
        unique_news = remove_duplicate_news(all_news)

        return unique_news[:10]
    return None

def sentiment_distribution(news):
    """Computes sentiment distribution of news articles"""
    if not news:
        return None

    sentiments = [article["sentiment"] for article in news]
    sentiment_counts = {
        "Positive": sentiments.count("Positive"),
        "Negative": sentiments.count("Negative"),
        "Neutral": sentiments.count("Neutral")
    }

    return sentiment_counts

def generate_comparative_analysis(news_articles):
    """
    Uses Gemini to generate a structured JSON comparative analysis.
    """
    # Constructing the prompt
    prompt = (
        "You are an AI that performs comparative news analysis. "
        "Given the following news articles, generate a structured JSON analysis "
        "highlighting how the news coverage differs.\n\n"
        "For each important comparison, include:\n"
        "- 'Comparison': A contrast between two or more news perspectives.\n"
        "- 'Impact': The effect of these differences on public perception or business implications.\n\n"
        "mention from which news articles comparion is made.\n\n"
        "Also mention news articles that are similar or share common keywords but dont consider name enities.\n\n"
        "Here are the articles:\n"
    )

    for idx, article in enumerate(news_articles, 1):
        prompt += f"{idx}. Title: {article['title']}\n"
        prompt += f"   Summary: {article['summary']}\n"

    prompt += (
        "Output the response in valid JSON format with the key 'Comparative Analysis' "
        "containing a list of comparisons."
    )

    # Call Gemini API
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    output=response.text
    json_start = output.find("{")
    result = output[json_start:-3]

    try:
        result = json.loads(result)
    except json.JSONDecodeError:
        result = {"Comparative Analysis": "Error parsing Gemini output."}

    return result

def generate_final_sentiment(news_articles):
    """
    Analyzes the overall sentiment of extracted news articles
    and generates a final sentiment summary using Gemini.
    """
    # Count sentiment occurrences
    sentiment_counts = Counter(article["sentiment"] for article in news_articles)

    prompt = (
        "You are an AI that performs sentiment analysis for company news. "
        "Based on the following sentiment distribution, generate a final overall sentiment summary.\n\n"
        f"Sentiment counts: {sentiment_counts}\n\n"
        "Output the response in a structured JSON format with a key 'Final Sentiment Analysis'."
    )

    # Call Gemini API
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    output = response.text
    json_start = output.find("{")
    result = output[json_start:-4]

    try:
        result = json.loads(result)
    except json.JSONDecodeError:
        result = {"Final Sentiment Analysis": "Error parsing Gemini output."}

    return result

def english_to_hindi_tts(english_text):
    # Translate English to Hindi
    translated_text = GoogleTranslator(source='en', target='hi').translate(english_text)

    # Convert Hindi text to speech
    tts = gTTS(text=translated_text, lang='hi')

    # Save to memory (without creating a file)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)  # Move to the start of the buffer

    return audio_buffer