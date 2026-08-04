import numpy as np
from typing import Optional, List

class DepressionPredictor:
    def __init__(self):
        # Weights for combining text and voice analysis
        self.text_weight = 0.6
        self.voice_weight = 0.4
        
        # Risk thresholds (CORRECTED: lower scores = lower risk)
        self.low_risk_threshold = 3.0      # 0-3: Low risk (happy/positive)
        self.moderate_risk_threshold = 5.0  # 3-5: Moderate risk (neutral)
        self.high_risk_threshold = 7.0      # 5-7: High risk (sad/concerning)
        # 7-10: Very high risk (very depressed)
    
    def predict_depression_risk(self, text_score: Optional[float], voice_score: Optional[float]) -> float:
        """
        Combine text and voice analysis scores to predict depression risk
        Returns a score from 0-10 where:
        0-3: Low risk (happy/positive content)
        3-5: Moderate risk (neutral content)
        5-7: High risk (sad/concerning content)
        7-10: Very high risk (very depressed content)
        """
        if text_score is None and voice_score is None:
            return 5.0  # Neutral score if no data
        
        # If only one score is available, use it with confidence adjustment
        if text_score is None:
            # Only voice data available - use it with slight conservative adjustment
            return max(0.0, min(10.0, voice_score))
        
        if voice_score is None:
            # Only text data available - use it with minimal adjustment
            return max(0.0, min(10.0, text_score))
        
        # Combine both scores using weighted average
        combined_score = (text_score * self.text_weight) + (voice_score * self.voice_weight)
        
        # Apply consistency check - if both scores are in similar ranges, increase confidence
        score_difference = abs(text_score - voice_score)
        if score_difference < 1.0:
            # Very consistent scores - no adjustment needed
            pass
        elif score_difference < 2.0:
            # Moderately consistent - slight adjustment toward neutral
            combined_score = combined_score * 0.95 + 5.0 * 0.05
        else:
            # Inconsistent scores - more conservative approach
            combined_score = combined_score * 0.85 + 5.0 * 0.15
        
        # Ensure score is within valid range
        return max(0.0, min(10.0, combined_score))
    
    def get_risk_level(self, score: float) -> str:
        """Convert numerical score to risk level category"""
        if score <= self.low_risk_threshold:
            return "Low Risk"
        elif score <= self.moderate_risk_threshold:
            return "Moderate Risk"
        elif score <= self.high_risk_threshold:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def get_recommendations(self, score: float) -> List[str]:
        """Get personalized recommendations based on risk score"""
        recommendations = []
        
        if score <= self.low_risk_threshold:
            # Low risk recommendations (0-3: Happy/positive indicators)
            recommendations.extend([
                "You're showing positive emotional indicators - that's wonderful!",
                "Continue maintaining the activities and habits that make you feel good",
                "Keep up healthy lifestyle practices like regular exercise and good sleep",
                "Stay connected with supportive friends and family",
                "Consider sharing your positive strategies with others who might benefit",
                "Remember to practice gratitude and celebrate small wins",
                "Maintain work-life balance and engage in hobbies you enjoy"
            ])
        
        elif score <= self.moderate_risk_threshold:
            # Moderate risk recommendations (3-5: Neutral indicators)
            recommendations.extend([
                "Your emotional indicators suggest you're in a neutral state",
                "Consider activities that typically boost your mood and energy",
                "Regular physical exercise can help improve emotional well-being",
                "Try mindfulness or meditation practices for emotional balance",
                "Maintain social connections and don't hesitate to reach out to friends",
                "Focus on consistent sleep schedule and healthy eating habits",
                "If you notice mood changes, monitor them and consider professional support"
            ])
        
        elif score <= self.high_risk_threshold:
            # High risk recommendations (5-7: Sad/concerning indicators)
            recommendations.extend([
                "Your responses indicate some concerning emotional patterns",
                "We strongly recommend speaking with a mental health professional",
                "Consider reaching out to trusted friends or family members for support",
                "Establish a daily routine to provide structure and stability",
                "Prioritize basic self-care: regular meals, sleep, and personal hygiene",
                "Avoid isolation - try to maintain some social contact each day",
                "Consider joining a support group or community activity",
                "Monitor your mood and symptoms regularly"
            ])
        
        else:
            # Very high risk recommendations (7-10: Very concerning indicators)
            recommendations.extend([
                "Your responses indicate serious emotional distress - please seek immediate help",
                "Contact a mental health crisis line or emergency services if needed",
                "Reach out to a mental health professional as soon as possible",
                "Ensure you have 24/7 access to support through crisis hotlines",
                "Stay with trusted friends or family members if possible",
                "Remove any means of self-harm from your environment",
                "Consider visiting an emergency room if you're having thoughts of self-harm",
                "Follow up with a mental health professional within 24-48 hours"
            ])
        
        # General recommendations for all risk levels
        recommendations.extend([
            "Remember that seeking help is a sign of strength, not weakness",
            "Mental health conditions are treatable with proper support",
            "Your feelings are valid and you deserve care and support"
        ])
        
        return recommendations
    
    def get_confidence_level(self, text_score: Optional[float], voice_score: Optional[float]) -> str:
        """Determine confidence level of the prediction"""
        if text_score is not None and voice_score is not None:
            # Both analyses available
            score_difference = abs(text_score - voice_score)
            if score_difference < 1.0:
                return "High Confidence - Consistent Results"
            elif score_difference < 2.0:
                return "Moderate Confidence - Generally Consistent"
            elif score_difference < 3.0:
                return "Low Confidence - Mixed Signals"
            else:
                return "Very Low Confidence - Conflicting Results"
        
        elif text_score is not None or voice_score is not None:
            # Only one analysis available
            return "Moderate Confidence - Single Analysis"
        
        else:
            # No analysis available
            return "No Confidence - No Data"
    
    def get_analysis_summary(self, text_score: Optional[float], voice_score: Optional[float]) -> dict:
        """Get comprehensive analysis summary"""
        combined_score = self.predict_depression_risk(text_score, voice_score)
        risk_level = self.get_risk_level(combined_score)
        confidence = self.get_confidence_level(text_score, voice_score)
        recommendations = self.get_recommendations(combined_score)
        
        # Determine emotional state description
        if combined_score <= 3.0:
            emotional_state = "Positive emotional indicators detected"
        elif combined_score <= 5.0:
            emotional_state = "Neutral emotional state"
        elif combined_score <= 7.0:
            emotional_state = "Concerning emotional indicators detected"
        else:
            emotional_state = "Serious emotional distress indicators detected"
        
        return {
            'combined_score': combined_score,
            'risk_level': risk_level,
            'emotional_state': emotional_state,
            'confidence': confidence,
            'recommendations': recommendations,
            'text_score': text_score,
            'voice_score': voice_score,
            'analysis_completeness': self._get_completeness_score(text_score, voice_score),
            'score_interpretation': self._get_score_interpretation(combined_score)
        }
    
    def _get_completeness_score(self, text_score: Optional[float], voice_score: Optional[float]) -> float:
        """Calculate how complete the analysis is (0-1 scale)"""
        if text_score is not None and voice_score is not None:
            return 1.0
        elif text_score is not None or voice_score is not None:
            return 0.6
        else:
            return 0.0
    
    def _get_score_interpretation(self, score: float) -> str:
        """Provide clear interpretation of the score"""
        if score <= 1.5:
            return "Very positive emotional indicators - excellent mental state"
        elif score <= 3.0:
            return "Positive emotional indicators - good mental health"
        elif score <= 4.0:
            return "Slightly positive to neutral emotional state"
        elif score <= 5.0:
            return "Neutral emotional state - monitor for changes"
        elif score <= 6.0:
            return "Some concerning emotional indicators - consider support"
        elif score <= 7.0:
            return "Concerning emotional patterns - seek professional help"
        elif score <= 8.5:
            return "Serious emotional distress - immediate professional help recommended"
        else:
            return "Severe emotional distress - seek emergency mental health support"
