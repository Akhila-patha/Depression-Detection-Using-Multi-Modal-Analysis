def get_mental_health_resources():
    """Return mental health resources and helplines"""
    return {
        "🆘 Crisis Helplines": [
            {
                "name": "National Suicide Prevention Lifeline",
                "description": "24/7 free and confidential support for people in distress",
                "contact": "988 or 1-800-273-8255",
                "website": "https://suicidepreventionlifeline.org"
            },
            {
                "name": "Crisis Text Line",
                "description": "Free, 24/7 crisis support via text message",
                "contact": "Text HOME to 741741",
                "website": "https://www.crisistextline.org"
            },
            {
                "name": "National Alliance on Mental Illness (NAMI)",
                "description": "Support and education for individuals and families",
                "contact": "1-800-950-NAMI (6264)",
                "website": "https://www.nami.org"
            }
        ],
        "🏥 Professional Help": [
            {
                "name": "Psychology Today",
                "description": "Find mental health professionals in your area",
                "website": "https://www.psychologytoday.com"
            },
            {
                "name": "SAMHSA Treatment Locator",
                "description": "Find treatment facilities and programs",
                "contact": "1-800-662-4357",
                "website": "https://findtreatment.samhsa.gov"
            },
            {
                "name": "Your Primary Care Doctor",
                "description": "Start with your family doctor for referrals and initial assessment"
            }
        ],
        "💪 Self-Help Resources": [
            {
                "name": "Headspace",
                "description": "Meditation and mindfulness app",
                "website": "https://www.headspace.com"
            },
            {
                "name": "Calm",
                "description": "Sleep, meditation, and relaxation app",
                "website": "https://www.calm.com"
            },
            {
                "name": "7 Cups",
                "description": "Free emotional support and online therapy",
                "website": "https://www.7cups.com"
            },
            {
                "name": "Mood Meter",
                "description": "Emotion regulation and mental health tracking app",
                "website": "https://www.moodmeterapp.com"
            }
        ],
        "📚 Educational Resources": [
            {
                "name": "National Institute of Mental Health (NIMH)",
                "description": "Comprehensive information about mental health conditions",
                "website": "https://www.nimh.nih.gov"
            },
            {
                "name": "Mental Health America",
                "description": "Mental health screening tools and resources",
                "website": "https://www.mentalhealthamerica.net"
            },
            {
                "name": "Anxiety and Depression Association of America",
                "description": "Resources for anxiety and depression support",
                "website": "https://adaa.org"
            }
        ]
    }

def get_risk_level_info(score):
    """Get risk level information including color and message with corrected mapping"""
    if score <= 3.0:
        return (
            "Low Risk",
            "#4CAF50",  # Green
            "Your responses suggest positive emotional indicators. You appear to be managing well and showing healthy emotional patterns. Continue with activities that support your wellbeing."
        )
    elif score <= 5.0:
        return (
            "Moderate Risk",
            "#FF9800",  # Orange
            "Your responses indicate neutral emotional patterns. Consider engaging in mood-boosting activities and maintaining healthy habits. Monitor your emotional state and don't hesitate to seek support if needed."
        )
    elif score <= 7.0:
        return (
            "High Risk",
            "#F44336",  # Red
            "Your responses suggest concerning emotional patterns that may indicate sadness or distress. We strongly recommend reaching out to a mental health professional for support and guidance."
        )
    else:
        return (
            "Very High Risk",
            "#D32F2F",  # Dark Red
            "Your responses indicate serious emotional distress that requires immediate attention. Please seek professional help right away or contact a crisis helpline for immediate support."
        )

def format_score_explanation(score_type, score):
    """Format explanation for different score types with corrected interpretations"""
    explanations = {
        "text": {
            "low": "Your text shows positive language patterns, emotional warmth, and optimistic expression - indicating good mental health.",
            "moderate": "Your text shows neutral emotional patterns with some mixed indicators. Consider engaging in mood-supporting activities.",
            "high": "Your text contains concerning language patterns that may indicate emotional distress, sadness, or difficulty coping."
        },
        "voice": {
            "low": "Your voice patterns suggest good energy levels, natural expression, and positive emotional tone - healthy indicators.",
            "moderate": "Your voice patterns show neutral characteristics with some mixed emotional indicators.",
            "high": "Your voice patterns may indicate low energy, subdued expression, or emotional distress that warrants attention."
        }
    }
    
    if score <= 3.0:
        level = "low"  # Low risk = positive indicators
    elif score <= 5.0:
        level = "moderate"  # Moderate risk = neutral
    else:
        level = "high"  # High risk = concerning indicators
    
    return explanations.get(score_type, {}).get(level, "Analysis completed.")

def get_emotional_keywords_summary():
    """Return summary of emotional keywords used in analysis"""
    return {
        "happiness_indicators": [
            "High intensity: ecstatic, euphoric, elated, thrilled, amazing, wonderful",
            "Moderate intensity: happy, joyful, cheerful, delighted, pleased, content",
            "Low intensity: okay, fine, alright, decent, calm, peaceful"
        ],
        "concern_indicators": [
            "High intensity: suicidal, hopeless, helpless, devastating, unbearable",
            "Moderate intensity: depressed, sad, unhappy, worried, anxious, exhausted",
            "Low intensity: concerned, unsure, uncomfortable, bothered, discouraged"
        ],
        "neutral_indicators": [
            "thinking, considering, working, studying, normal, routine, typical"
        ]
    }

def validate_score_range(score):
    """Ensure score is within valid range and provide warning if not"""
    if score < 0 or score > 10:
        print(f"Warning: Score {score} is outside valid range [0-10]")
        return max(0, min(10, score))
    return score

def get_confidence_description(text_score, voice_score):
    """Get description of analysis confidence based on available data"""
    if text_score is not None and voice_score is not None:
        difference = abs(text_score - voice_score)
        if difference < 1.0:
            return "High confidence - both text and voice analysis show consistent results"
        elif difference < 2.0:
            return "Good confidence - text and voice analysis show generally consistent patterns"
        else:
            return "Moderate confidence - text and voice analysis show some conflicting patterns"
    elif text_score is not None:
        return "Moderate confidence - analysis based on text input only"
    elif voice_score is not None:
        return "Moderate confidence - analysis based on voice input only"
    else:
        return "No confidence - insufficient data for analysis"
