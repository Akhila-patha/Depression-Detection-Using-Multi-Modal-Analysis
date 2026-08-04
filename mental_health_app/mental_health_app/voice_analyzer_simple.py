import numpy as np
import warnings
import wave
import tempfile
import os
from pydub import AudioSegment
warnings.filterwarnings('ignore')

class VoiceAnalyzer:
    def __init__(self):
        self.sample_rate = 22050
        
    def analyze_voice(self, audio_data=None, audio_file_path=None):
        """
        Analyze voice patterns and return a depression risk score (0-10)
        0-3: Low risk (happy/energetic voice)
        3-5: Moderate risk (neutral voice)
        5-7: High risk (sad/low energy voice)
        7-10: Very high risk (very depressed voice patterns)
        """
        try:
            if audio_data is not None:
                # Convert audio data to temporary file for analysis
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_data)
                    temp_path = tmp_file.name
                
                score = self._analyze_audio_file(temp_path)
                
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
                return score
                
            elif audio_file_path:
                return self._analyze_audio_file(audio_file_path)
            else:
                return 5.0  # Neutral score if no audio provided
                
        except Exception as e:
            print(f"Error analyzing voice: {e}")
            return 5.0  # Return neutral score on error
    
    def _analyze_audio_file(self, file_path):
        """Analyze audio file and return depression risk score"""
        try:
            # Load audio using pydub
            audio = AudioSegment.from_file(file_path)
            
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples())
            
            # If stereo, convert to mono
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
                samples = samples.mean(axis=1)
            
            # Normalize
            if np.max(np.abs(samples)) > 0:
                samples = samples / np.max(np.abs(samples))
            
            # Basic audio analysis
            duration = len(samples) / audio.frame_rate
            
            # Energy analysis (vocal energy indicates emotional state)
            energy = np.mean(samples ** 2)
            
            # Speech rate estimation (zero crossings)
            zero_crossings = np.sum(np.diff(np.signbit(samples)))
            speech_rate = zero_crossings / (duration * 2) if duration > 0 else 0
            
            # Pause detection (silence regions)
            silence_threshold = 0.01
            silent_samples = np.abs(samples) < silence_threshold
            pause_ratio = np.sum(silent_samples) / len(samples) if len(samples) > 0 else 0
            
            # Pitch variation estimation (simplified)
            pitch_variation = np.std(samples) / np.mean(np.abs(samples)) if np.mean(np.abs(samples)) > 0 else 0
            
            # Calculate risk score based on emotional indicators in voice
            risk_score = self._calculate_emotional_risk_score(
                energy, speech_rate, pause_ratio, pitch_variation, duration
            )
            
            return max(0, min(10, risk_score))
            
        except Exception as e:
            print(f"Error in audio file analysis: {e}")
            return 5.0
    
    def _calculate_emotional_risk_score(self, energy, speech_rate, pause_ratio, pitch_variation, duration):
        """Calculate depression risk score from audio features with proper emotional mapping"""
        
        # Start with neutral baseline
        base_score = 5.0
        
        # Energy Level Analysis (Happy voices are typically more energetic)
        if energy > 0.20:  # Very high energy - indicates very positive/happy state
            base_score -= 4.5  # Significantly reduce risk (0.5 range)
        elif energy > 0.12:  # High energy - indicates positive/happy state
            base_score -= 3.5  # Significantly reduce risk (1.5 range)
        elif energy > 0.06:  # Moderate energy - healthy range
            base_score -= 2.0  # Reduce risk (3.0 range)
        elif energy > 0.03:  # Low-moderate energy - neutral to concerning
            base_score += 0.5  # Slight increase in risk (5.5 range)
        elif energy > 0.01:  # Low energy - concerning
            base_score += 2.0  # Increase risk (7.0 range)
        else:  # Very low energy - very concerning
            base_score += 3.5  # Significantly increase risk (8.5 range)
        
        # Speech Rate Analysis (Happy people tend to speak at normal to slightly faster rates)
        if 140 <= speech_rate <= 280:  # Optimal range - engaged and animated
            base_score -= 1.5  # Reduce risk more aggressively
        elif 100 <= speech_rate < 140:  # Moderate pace - normal range
            base_score -= 0.5  # Slight reduction for normal pace
        elif 50 <= speech_rate < 100:  # Slow speech - may indicate sadness
            base_score += 1.5  # Increase risk
        elif speech_rate < 50:  # Very slow speech - concerning
            base_score += 2.5  # Significantly increase risk
        elif speech_rate > 300:  # Very fast speech - may indicate anxiety
            base_score += 1.0  # Moderate increase
        
        # Pause Analysis (Excessive pauses may indicate hesitation/sadness)
        if pause_ratio < 0.25:  # Very few pauses - confident, flowing speech
            base_score -= 1.0  # Reduce risk more significantly
        elif pause_ratio < 0.45:  # Normal pause frequency
            base_score -= 0.2  # Slight reduction for normal pauses
        elif pause_ratio < 0.65:  # Moderate pauses - slight concern
            base_score += 0.5  # Slight increase
        elif pause_ratio < 0.8:  # Many pauses - concerning
            base_score += 1.5  # Increase risk
        else:  # Excessive pauses - very concerning
            base_score += 2.5  # Significantly increase risk
        
        # Pitch Variation Analysis (Happy voices typically have more variation)
        if pitch_variation > 0.3:  # High variation - expressive and animated
            base_score -= 1.5  # Reduce risk
        elif pitch_variation > 0.15:  # Moderate variation - normal expression
            base_score -= 0.5  # Slight reduction
        elif pitch_variation > 0.05:  # Low variation - monotone tendency
            base_score += 1.0  # Increase risk
        else:  # Very low variation - flat/monotone (concerning)
            base_score += 2.0  # Significantly increase risk
        
        # Duration considerations (very short clips are less reliable)
        if duration < 5:
            base_score += 0.5  # Less confident for short clips
        elif duration < 2:
            base_score = 5.0  # Very conservative for very short clips
        
        # Ensure the score stays within bounds
        return max(0, min(10, base_score))
    
    def _get_emotional_tone_description(self, energy, speech_rate, pause_ratio, pitch_variation):
        """Generate emotional tone descriptions based on voice features"""
        
        # Energy descriptions
        if energy > 0.15:
            energy_desc = "High energy - positive and animated"
        elif energy > 0.08:
            energy_desc = "Moderate energy - engaged and alert"
        elif energy > 0.03:
            energy_desc = "Low-moderate energy - calm but possibly subdued"
        elif energy > 0.01:
            energy_desc = "Low energy - subdued and flat"
        else:
            energy_desc = "Very low energy - concerning lack of vocal vitality"
        
        # Pace descriptions
        if 150 <= speech_rate <= 250:
            pace_desc = "Optimal pace - clear and engaging"
        elif 100 <= speech_rate < 150:
            pace_desc = "Moderate pace - normal conversational speed"
        elif 50 <= speech_rate < 100:
            pace_desc = "Slow pace - hesitant or deliberate speech"
        elif speech_rate < 50:
            pace_desc = "Very slow pace - significantly delayed speech"
        elif speech_rate > 300:
            pace_desc = "Fast pace - possibly anxious or excited"
        else:
            pace_desc = "Unusual pace - irregular speech pattern"
        
        # Quality descriptions
        if pitch_variation > 0.3 and pause_ratio < 0.4:
            quality_desc = "Bright and expressive tone"
        elif pitch_variation > 0.15 and pause_ratio < 0.6:
            quality_desc = "Normal emotional expression"
        elif pitch_variation < 0.05 or pause_ratio > 0.7:
            quality_desc = "Monotone with frequent pauses"
        else:
            quality_desc = "Mixed emotional indicators"
        
        return {
            'energy_description': energy_desc,
            'pace_description': pace_desc,
            'quality_description': quality_desc
        }
    
    def get_detailed_analysis(self, audio_data=None, audio_file_path=None):
        """Get detailed analysis results"""
        try:
            if audio_data is not None:
                # Convert audio data to temporary file for analysis
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_data)
                    temp_path = tmp_file.name
                
                result = self._get_detailed_analysis_from_file(temp_path)
                
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
                return result
                
            elif audio_file_path:
                return self._get_detailed_analysis_from_file(audio_file_path)
            else:
                return self._get_default_analysis()
                
        except Exception as e:
            print(f"Error in detailed analysis: {e}")
            return self._get_default_analysis()
    
    def _get_detailed_analysis_from_file(self, file_path):
        """Get detailed analysis from audio file"""
        try:
            audio = AudioSegment.from_file(file_path)
            samples = np.array(audio.get_array_of_samples())
            
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
                samples = samples.mean(axis=1)
            
            if np.max(np.abs(samples)) > 0:
                samples = samples / np.max(np.abs(samples))
            
            duration = len(samples) / audio.frame_rate
            
            # Calculate features
            energy = np.mean(samples ** 2)
            zero_crossings = np.sum(np.diff(np.signbit(samples)))
            speech_rate = zero_crossings / (duration * 2) if duration > 0 else 0
            
            silence_threshold = 0.01
            silent_samples = np.abs(samples) < silence_threshold
            pause_ratio = np.sum(silent_samples) / len(samples) if len(samples) > 0 else 0
            
            # Estimate pitch variation (simplified)
            pitch_variation = np.std(samples) / np.mean(np.abs(samples)) if np.mean(np.abs(samples)) > 0 else 0
            
            # Get emotional tone descriptions
            emotional_tone = self._get_emotional_tone_description(energy, speech_rate, pause_ratio, pitch_variation)
            
            # Determine risk indicators
            risk_indicators = []
            if energy < 0.03:
                risk_indicators.append("Low vocal energy detected")
            if speech_rate < 100:
                risk_indicators.append("Slow speech patterns")
            if pause_ratio > 0.6:
                risk_indicators.append("Excessive pause frequency")
            if pitch_variation < 0.1:
                risk_indicators.append("Monotone speech quality")
            if energy > 0.15 and pitch_variation > 0.3:
                risk_indicators.append("Positive emotional indicators detected")
            if not risk_indicators:
                risk_indicators.append("Normal voice patterns detected")
            
            return {
                'pitch_variation': min(1.0, pitch_variation),
                'energy_level': min(1.0, energy * 10),
                'speech_rate': min(10.0, speech_rate / 50),
                'pause_frequency': min(1.0, pause_ratio),
                'duration': duration,
                'avg_pause_duration': pause_ratio * duration,
                'emotional_tone': emotional_tone,
                'risk_indicators': "; ".join(risk_indicators)
            }
            
        except Exception as e:
            print(f"Error in detailed file analysis: {e}")
            return self._get_default_analysis()
    
    def _get_default_analysis(self):
        """Return default analysis values"""
        return {
            'pitch_variation': 0.15,
            'energy_level': 0.05,
            'speech_rate': 3.2,
            'pause_frequency': 0.5,
            'duration': 5.0,
            'avg_pause_duration': 0.5,
            'emotional_tone': {
                'energy_description': 'Moderate energy - normal range',
                'pace_description': 'Normal pace - conversational speed',
                'quality_description': 'Normal emotional expression'
            },
            'risk_indicators': 'Default analysis - no audio provided'
        }
