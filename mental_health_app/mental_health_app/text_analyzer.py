import nltk
import spacy
import pandas as pd
import numpy as np
from collections import Counter
import re
from textblob import TextBlob
import streamlit as st

class TextAnalyzer:
    def __init__(self):
        self.setup_nltk()
        self.setup_spacy()
        
        # Expanded happiness keywords with emotional intensity
        self.happiness_keywords = {
            'high_intensity': [
                'ecstatic', 'euphoric', 'elated', 'overjoyed', 'thrilled', 'exhilarated',
                'blissful', 'rapturous', 'jubilant', 'exuberant', 'fantastic', 'amazing',
                'wonderful', 'incredible', 'magnificent', 'spectacular', 'brilliant',
                'phenomenal', 'outstanding', 'excellent', 'superb', 'marvelous', 'perfect',
                'divine', 'heavenly', 'magical', 'extraordinary'
            ],
            'moderate_intensity': [
                'happy', 'joyful', 'cheerful', 'delighted', 'pleased', 'content',
                'satisfied', 'glad', 'upbeat', 'positive', 'optimistic', 'hopeful',
                'grateful', 'blessed', 'good', 'great', 'nice', 'pleasant',
                'enjoyable', 'fun', 'excited', 'enthusiastic', 'motivated', 'light',
                'bright', 'fresh', 'better', 'beautiful', 'lovely', 'sweet', 'warm',
                'comfortable', 'relaxed', 'smile', 'laugh', 'dance', 'sing', 'aligned',
                'right', 'love', 'loved', 'appreciation', 'grateful', 'thankful'
            ],
            'low_intensity': [
                'okay', 'fine', 'alright', 'decent', 'reasonable', 'adequate',
                'acceptable', 'tolerable', 'manageable', 'stable', 'calm', 'peaceful',
                'gentle', 'soft', 'smooth', 'easy', 'simple', 'clear'
            ]
        }
        
        # Expanded sadness/depression keywords with emotional intensity
        self.sadness_keywords = {
            'high_intensity': [
                'suicidal', 'hopeless', 'helpless', 'worthless', 'devastating', 'crushing',
                'unbearable', 'agonizing', 'tormented', 'tortured', 'destroyed', 'shattered',
                'broken', 'ruined', 'doomed', 'damned', 'nightmare', 'horror', 'terrible',
                'awful', 'dreadful', 'miserable', 'wretched', 'pathetic', 'useless'
            ],
            'moderate_intensity': [
                'depressed', 'sad', 'unhappy', 'miserable', 'down', 'blue', 'gloomy',
                'melancholy', 'sorrowful', 'grieving', 'mourning', 'disappointed',
                'frustrated', 'upset', 'troubled', 'worried', 'anxious', 'stressed',
                'overwhelmed', 'exhausted', 'tired', 'drained', 'empty', 'numb',
                'lonely', 'isolated', 'disconnected', 'lost', 'confused', 'scared'
            ],
            'low_intensity': [
                'concerned', 'unsure', 'uncertain', 'hesitant', 'doubtful', 'questioning',
                'uncomfortable', 'uneasy', 'restless', 'bothered', 'annoyed', 'irritated',
                'disappointed', 'let down', 'discouraged', 'unmotivated', 'bored'
            ]
        }
        
        # Neutral emotional indicators
        self.neutral_keywords = [
            'thinking', 'considering', 'wondering', 'reflecting', 'pondering',
            'working', 'studying', 'learning', 'reading', 'writing', 'planning',
            'routine', 'normal', 'usual', 'regular', 'typical', 'average',
            'day', 'time', 'moment', 'today', 'yesterday', 'tomorrow'
        ]
        
    def setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
    
    def setup_spacy(self):
        """Setup spaCy model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            st.warning("spaCy English model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
    
    def analyze_text(self, text):
        """
        Analyze text and return a depression risk score (0-10)
        0-3: Low risk (happy/positive content)
        3-5: Moderate risk (neutral content)
        5-7: High risk (sad/concerning content)
        7-10: Very high risk (very depressed content)
        """
        if not text or len(text.strip()) < 10:
            return 5.0  # Neutral score for insufficient text
        
        # Clean and preprocess text
        text = self.preprocess_text(text)
        
        # Multiple analysis approaches
        sentiment_score = self.analyze_sentiment(text)
        keyword_score = self.analyze_keywords_with_intensity(text)
        emotional_intensity = self.calculate_emotional_intensity(text)
        linguistic_score = self.analyze_linguistic_patterns(text)
        
        # Combine scores with proper weighting
        # Sentiment and keywords are most important for emotional assessment
        final_score = (
            sentiment_score * 0.4 +      # Primary sentiment analysis
            keyword_score * 0.4 +        # Emotional keyword analysis
            emotional_intensity * 0.15 + # Emotional intensity factor
            linguistic_score * 0.05      # Linguistic patterns (minimal weight)
        )
        
        # Ensure score is within range [0, 10]
        return max(0, min(10, final_score))
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation for sentiment analysis
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        
        return text.strip()
    
    def analyze_sentiment(self, text):
        """Analyze sentiment and convert to risk score"""
        from nltk.sentiment import SentimentIntensityAnalyzer
        
        # TextBlob sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        
        # VADER sentiment
        sia = SentimentIntensityAnalyzer()
        vader_scores = sia.polarity_scores(text)
        compound_score = vader_scores['compound']  # -1 to 1
        
        # Combine sentiment scores
        avg_sentiment = (polarity + compound_score) / 2
        
        # Convert sentiment to risk score (FIXED MAPPING)
        # Positive sentiment = Low risk, Negative sentiment = High risk
        if avg_sentiment >= 0.2:  # Very positive
            risk_score = max(0, 2.5 - (avg_sentiment * 5))  # 0-1.5 range for very positive
        elif avg_sentiment >= 0.05:  # Moderately positive  
            risk_score = max(0, 3.5 - (avg_sentiment * 7))  # 1.5-3.2 range
        elif avg_sentiment >= -0.05:  # Neutral
            risk_score = 4.5 + (avg_sentiment * 3)  # 4.3-4.7 range
        elif avg_sentiment >= -0.2:  # Moderately negative
            risk_score = 6.0 + (abs(avg_sentiment) * 8)  # 5.6-7.6 range
        else:  # Very negative
            risk_score = min(10, 8.0 + (abs(avg_sentiment) * 5))  # 8-10 range
        
        return max(0, min(10, risk_score))
    
    def analyze_keywords_with_intensity(self, text):
        """Analyze emotional keywords with intensity weighting"""
        words = text.lower().split()
        
        if len(words) == 0:
            return 5.0
        
        # Count keywords by intensity
        happiness_scores = {
            'high': sum(1 for word in words if any(hw in word for hw in self.happiness_keywords['high_intensity'])),
            'moderate': sum(1 for word in words if any(hw in word for hw in self.happiness_keywords['moderate_intensity'])),
            'low': sum(1 for word in words if any(hw in word for hw in self.happiness_keywords['low_intensity']))
        }
        
        sadness_scores = {
            'high': sum(1 for word in words if any(sw in word for sw in self.sadness_keywords['high_intensity'])),
            'moderate': sum(1 for word in words if any(sw in word for sw in self.sadness_keywords['moderate_intensity'])),
            'low': sum(1 for word in words if any(sw in word for sw in self.sadness_keywords['low_intensity']))
        }
        
        neutral_count = sum(1 for word in words if any(nw in word for nw in self.neutral_keywords))
        
        # Calculate weighted emotional scores
        happiness_total = (happiness_scores['high'] * 3 + 
                          happiness_scores['moderate'] * 2 + 
                          happiness_scores['low'] * 1)
        
        sadness_total = (sadness_scores['high'] * 3 + 
                        sadness_scores['moderate'] * 2 + 
                        sadness_scores['low'] * 1)
        
        # Calculate base risk score
        if happiness_total > sadness_total:
            # More happiness indicators - low risk
            ratio = happiness_total / len(words)
            base_score = max(0, 3 - (ratio * 15))  # Strong happiness = very low risk
        elif sadness_total > happiness_total:
            # More sadness indicators - high risk
            ratio = sadness_total / len(words)
            base_score = min(10, 6 + (ratio * 20))  # Strong sadness = high risk
        else:
            # Balanced or neutral
            base_score = 4.5 + (neutral_count / len(words))  # Neutral range
        
        return max(0, min(10, base_score))
    
    def calculate_emotional_intensity(self, text):
        """Calculate overall emotional intensity"""
        words = text.lower().split()
        
        # Emotional intensifiers
        intensifiers = ['very', 'extremely', 'incredibly', 'absolutely', 'completely',
                       'totally', 'really', 'so', 'quite', 'rather', 'pretty']
        
        # Emotional diminishers
        diminishers = ['somewhat', 'slightly', 'a bit', 'a little', 'kind of',
                      'sort of', 'maybe', 'perhaps', 'possibly']
        
        intensifier_count = sum(1 for word in words if word in intensifiers)
        diminisher_count = sum(1 for word in words if word in diminishers)
        
        # Calculate intensity factor
        if len(words) == 0:
            return 5.0
        
        intensity_factor = (intensifier_count - diminisher_count) / len(words)
        
        # Convert to risk score (intense emotions can indicate either very good or very bad states)
        if intensity_factor > 0.1:  # High intensity
            return 6.0  # Moderate-high risk due to emotional volatility
        elif intensity_factor < -0.1:  # Low intensity (diminished emotions)
            return 4.0  # Moderate risk
        else:
            return 5.0  # Neutral intensity
    
    def analyze_linguistic_patterns(self, text):
        """Analyze linguistic patterns with minimal impact"""
        doc = self.nlp(text)
        
        # Excessive first-person focus (only very excessive is concerning)
        first_person_count = sum(1 for token in doc if token.text.lower() in ['i', 'me', 'my', 'myself'])
        
        # Negative constructions
        negative_words = ['no', 'not', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'nor']
        negative_count = sum(1 for token in doc if token.text.lower() in negative_words)
        
        # Calculate ratios
        total_words = len([token for token in doc if token.is_alpha])
        if total_words == 0:
            return 5.0
        
        first_person_ratio = first_person_count / total_words
        negative_ratio = negative_count / total_words
        
        # Start with neutral score
        risk_score = 5.0
        
        # Only penalize very excessive patterns
        if first_person_ratio > 0.4:  # More than 40% first-person
            risk_score += (first_person_ratio - 0.4) * 5
        
        if negative_ratio > 0.3:  # More than 30% negative words
            risk_score += (negative_ratio - 0.3) * 5
        
        return min(10, max(0, risk_score))
    
    def get_detailed_analysis(self, text):
        """Get detailed analysis results"""
        blob = TextBlob(text)
        doc = self.nlp(text)
        words = text.lower().split()
        
        # Find specific emotional keywords
        happy_keywords = []
        sad_keywords = []
        neutral_indicators = []
        
        for word in words:
            # Check happiness keywords
            for intensity_level, keyword_list in self.happiness_keywords.items():
                if any(hw in word for hw in keyword_list):
                    happy_keywords.append(f"{word} ({intensity_level})")
            
            # Check sadness keywords
            for intensity_level, keyword_list in self.sadness_keywords.items():
                if any(sw in word for sw in keyword_list):
                    sad_keywords.append(f"{word} ({intensity_level})")
            
            # Check neutral keywords
            if any(nw in word for nw in self.neutral_keywords):
                neutral_indicators.append(word)
        
        # Sentiment breakdown
        sentiment_breakdown = {
            'Positive': max(0, blob.sentiment.polarity),
            'Negative': max(0, -blob.sentiment.polarity),
            'Neutral': 1 - abs(blob.sentiment.polarity)
        }
        
        # Overall sentiment classification
        if blob.sentiment.polarity > 0.1:
            overall_sentiment = "Positive"
        elif blob.sentiment.polarity < -0.1:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"
        
        # Calculate emotional intensity
        emotional_intensity = self.calculate_emotional_intensity(text)
        
        # Determine risk factors
        risk_factors = []
        if sad_keywords:
            risk_factors.append("Concerning emotional language detected")
        if len([k for k in sad_keywords if 'high_intensity' in k]) > 0:
            risk_factors.append("High-intensity negative emotions")
        if not happy_keywords and not neutral_indicators:
            risk_factors.append("Absence of positive indicators")
        if not risk_factors:
            risk_factors.append("No significant risk factors detected")
        
        # Key phrases extraction
        key_phrases = []
        for chunk in doc.noun_chunks:
            if len(chunk.text.strip()) > 2:
                key_phrases.append(chunk.text.strip())
        
        return {
            'sentiment_breakdown': sentiment_breakdown,
            'overall_sentiment': overall_sentiment,
            'emotional_intensity': emotional_intensity,
            'happy_keywords': list(set(happy_keywords)),
            'sad_keywords': list(set(sad_keywords)),
            'neutral_indicators': list(set(neutral_indicators)),
            'risk_factors': "; ".join(risk_factors),
            'key_phrases': key_phrases[:10],  # Top 10 phrases
            'word_count': len(words),
            'subjectivity': blob.sentiment.subjectivity
        }
