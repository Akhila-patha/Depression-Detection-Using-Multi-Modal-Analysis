import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
import numpy as np
from text_analyzer import TextAnalyzer
from voice_analyzer_simple import VoiceAnalyzer
from depression_predictor import DepressionPredictor
from utils import get_mental_health_resources, get_risk_level_info

def main():
    st.set_page_config(
        page_title="Mental health Assessment Tool",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize analyzers
    if 'text_analyzer' not in st.session_state:
        st.session_state.text_analyzer = TextAnalyzer()
    if 'voice_analyzer' not in st.session_state:
        st.session_state.voice_analyzer = VoiceAnalyzer()
    if 'depression_predictor' not in st.session_state:
        st.session_state.depression_predictor = DepressionPredictor()
    
    # Header
    st.title("🧠 Mental health Assessment Tool")
    st.markdown("---")
    
    # Important disclaimer
    st.warning("""
    ⚠️ **Important Disclaimer**: This tool is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. If you're experiencing mental health concerns, please consult with a qualified healthcare professional.
    """)
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose a section:",
            ["Assessment", "Resources", "About"]
        )
        
    
    if page == "Assessment":
        show_assessment_page()
    elif page == "Resources":
        show_resources_page()
    elif page == "Download Project":
        show_download_page()
    else:
        show_about_page()

def show_assessment_page():
    st.header("Mental Health Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Text Analysis")
        st.markdown("Please share your thoughts and feelings:")
        
        text_input = st.text_area(
            "How are you feeling today?",
            height=200,
            placeholder="Share your thoughts, feelings, or experiences here..."
        )
        
        text_score = None
        if text_input:
            with st.spinner("Analyzing text..."):
                text_score = st.session_state.text_analyzer.analyze_text(text_input)
            
            st.success(f"Text analysis completed! Risk Score: {text_score:.2f}")
            
            # Display text analysis details
            with st.expander("View Text Analysis Details"):
                sentiment_data = st.session_state.text_analyzer.get_detailed_analysis(text_input)
                
                # Show detected emotional indicators
                st.write("**Emotional Indicators Found:**")
                if sentiment_data['happy_keywords']:
                    st.write("😊 **Happy Keywords:**", ", ".join(sentiment_data['happy_keywords']))
                if sentiment_data['sad_keywords']:
                    st.write("😢 **Concerning Keywords:**", ", ".join(sentiment_data['sad_keywords']))
                if sentiment_data['neutral_indicators']:
                    st.write("😐 **Neutral Indicators:**", ", ".join(sentiment_data['neutral_indicators']))
                
                # Sentiment breakdown
                fig_sentiment = px.bar(
                    x=list(sentiment_data['sentiment_breakdown'].keys()),
                    y=list(sentiment_data['sentiment_breakdown'].values()),
                    title="Sentiment Breakdown",
                    color=list(sentiment_data['sentiment_breakdown'].keys()),
                    color_discrete_map={
                        'Positive': '#4CAF50',
                        'Negative': '#F44336',
                        'Neutral': '#FF9800'
                    }
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)
                
                # Scoring explanation
                st.write("**Scoring Details:**")
                st.write(f"- Overall Sentiment: {sentiment_data['overall_sentiment']}")
                st.write(f"- Emotional Intensity: {sentiment_data['emotional_intensity']:.2f}")
                st.write(f"- Risk Factors: {sentiment_data['risk_factors']}")
                
                # Key phrases
                if sentiment_data['key_phrases']:
                    st.write("**Key Phrases Detected:**")
                    for phrase in sentiment_data['key_phrases']:
                        st.write(f"- {phrase}")
    
    with col2:
        st.subheader("🎤 Voice Analysis")
        st.markdown("Upload an audio file to analyze speech patterns:")
        
        voice_score = None
        
        # Simple instructions for voice recording
        st.info("""
        📱 **How to record your voice:**
        1. Use your phone's voice recorder app
        2. Record yourself speaking for 30-60 seconds
        3. Save the recording as a file
        4. Upload it using the button below
        """)
        
        # Audio file upload
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'ogg', 'm4a', 'webm'],
            help="Upload a voice recording in WAV, MP3, OGG, M4A, or WebM format"
        )
        
        if uploaded_file is not None:
            # Show file details
            st.write(f"**File name:** {uploaded_file.name}")
            st.write(f"**File size:** {uploaded_file.size / 1024:.1f} KB")
            
            # Play audio file
            st.audio(uploaded_file, format='audio/wav')
            
            if st.button("Analyze Voice", type="primary"):
                with st.spinner("Analyzing voice patterns..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        temp_path = tmp_file.name
                    
                    try:
                        voice_score = st.session_state.voice_analyzer.analyze_voice(audio_file_path=temp_path)
                        st.success(f"Voice analysis completed! Risk Score: {voice_score:.2f}")
                        
                        # Display voice analysis details
                        with st.expander("View Voice Analysis Details"):
                            voice_data = st.session_state.voice_analyzer.get_detailed_analysis(audio_file_path=temp_path)
                            
                            # Voice emotional indicators
                            st.write("**Emotional Tone Analysis:**")
                            st.write(f"- Energy Level: {voice_data['emotional_tone']['energy_description']}")
                            st.write(f"- Speech Pattern: {voice_data['emotional_tone']['pace_description']}")
                            st.write(f"- Voice Quality: {voice_data['emotional_tone']['quality_description']}")
                            
                            # Voice features
                            features_df = pd.DataFrame({
                                'Feature': ['Energy Level', 'Speech Rate', 'Pause Frequency', 'Pitch Variation'],
                                'Value': [
                                    voice_data['energy_level'],
                                    voice_data['speech_rate'],
                                    voice_data['pause_frequency'],
                                    voice_data['pitch_variation']
                                ]
                            })
                            
                            fig_voice = px.bar(
                                features_df,
                                x='Feature',
                                y='Value',
                                title="Voice Pattern Analysis",
                                color='Feature'
                            )
                            st.plotly_chart(fig_voice, use_container_width=True)
                            
                            # Additional details
                            st.write("**Analysis Details:**")
                            st.write(f"- Duration: {voice_data['duration']:.1f} seconds")
                            st.write(f"- Energy Level: {voice_data['energy_level']:.3f}")
                            st.write(f"- Speech Rate: {voice_data['speech_rate']:.2f}")
                            st.write(f"- Pause Frequency: {voice_data['pause_frequency']:.2f}")
                            st.write(f"- Risk Factors: {voice_data['risk_indicators']}")
                            
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
        
        # Demo voice analysis option
        st.markdown("---")
        st.markdown("**Don't have an audio file? Try a demo:**")
        demo_col1, demo_col2 = st.columns(2)
        
        with demo_col1:
            if st.button("😊 Happy Voice Demo", help="Simulate analysis of happy/energetic voice"):
                voice_score = 1.5  # Low risk for happy voice
                st.success(f"Happy voice demo completed! Risk Score: {voice_score:.2f}")
                
                with st.expander("View Happy Voice Demo Details"):
                    demo_data = {
                        'pitch_variation': 0.45,
                        'energy_level': 0.30,  # High energy for visualization
                        'speech_rate': 5.8,
                        'pause_frequency': 0.2,
                        'duration': 15.0,
                        'emotional_tone': {
                            'energy_description': 'High energy - positive and animated',
                            'pace_description': 'Optimal pace - clear and engaging',
                            'quality_description': 'Bright and expressive tone'
                        },
                        'risk_indicators': 'Positive emotional indicators detected'
                    }
                    
                    st.write("**Emotional Tone Analysis:**")
                    st.write(f"- Energy Level: {demo_data['emotional_tone']['energy_description']}")
                    st.write(f"- Speech Pattern: {demo_data['emotional_tone']['pace_description']}")
                    st.write(f"- Voice Quality: {demo_data['emotional_tone']['quality_description']}")
                    
                    features_df = pd.DataFrame({
                        'Feature': ['Energy Level', 'Speech Rate', 'Pause Frequency', 'Pitch Variation'],
                        'Value': [
                            demo_data['energy_level'],
                            demo_data['speech_rate'],
                            demo_data['pause_frequency'],
                            demo_data['pitch_variation']
                        ]
                    })
                    
                    fig_voice = px.bar(
                        features_df,
                        x='Feature',
                        y='Value',
                        title="Happy Voice Pattern Analysis",
                        color='Feature'
                    )
                    st.plotly_chart(fig_voice, use_container_width=True)
        
        with demo_col2:
            if st.button("😢 Sad Voice Demo", help="Simulate analysis of sad/low energy voice"):
                voice_score = 7.2  # High risk for sad voice
                st.warning(f"Sad voice demo completed! Risk Score: {voice_score:.2f}")
                
                with st.expander("View Sad Voice Demo Details"):
                    demo_data = {
                        'pitch_variation': 0.12,
                        'energy_level': 0.08,
                        'speech_rate': 2.1,
                        'pause_frequency': 0.75,
                        'duration': 15.0,
                        'emotional_tone': {
                            'energy_description': 'Low energy - subdued and flat',
                            'pace_description': 'Slow pace - hesitant speech',
                            'quality_description': 'Monotone with frequent pauses'
                        },
                        'risk_indicators': 'Concerning emotional patterns detected'
                    }
                    
                    st.write("**Emotional Tone Analysis:**")
                    st.write(f"- Energy Level: {demo_data['emotional_tone']['energy_description']}")
                    st.write(f"- Speech Pattern: {demo_data['emotional_tone']['pace_description']}")
                    st.write(f"- Voice Quality: {demo_data['emotional_tone']['quality_description']}")
                    
                    features_df = pd.DataFrame({
                        'Feature': ['Energy Level', 'Speech Rate', 'Pause Frequency', 'Pitch Variation'],
                        'Value': [
                            demo_data['energy_level'],
                            demo_data['speech_rate'],
                            demo_data['pause_frequency'],
                            demo_data['pitch_variation']
                        ]
                    })
                    
                    fig_voice = px.bar(
                        features_df,
                        x='Feature',
                        y='Value',
                        title="Sad Voice Pattern Analysis",
                        color='Feature'
                    )
                    st.plotly_chart(fig_voice, use_container_width=True)
    
    # Combined analysis
    if text_score is not None or voice_score is not None:
        st.markdown("---")
        st.header("🎯 Combined Assessment Results")
        
        # Calculate combined score
        combined_score = st.session_state.depression_predictor.predict_depression_risk(
            text_score, voice_score
        )
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Text Analysis Risk Score",
                f"{text_score:.2f}" if text_score is not None else "N/A",
                help="Lower scores (0-3) indicate positive content, higher scores (7-10) indicate concerning content"
            )
        
        with col2:
            st.metric(
                "Voice Analysis Risk Score",
                f"{voice_score:.2f}" if voice_score is not None else "N/A",
                help="Lower scores (0-3) indicate positive tone, higher scores (7-10) indicate concerning patterns"
            )
        
        with col3:
            st.metric(
                "Combined Risk Assessment",
                f"{combined_score:.2f}",
                help="Overall risk assessment: 0-3 (Low), 3-5 (Moderate), 5-7 (High), 7-10 (Very High)"
            )
        
        # Risk level indicator
        risk_level, color, message = get_risk_level_info(combined_score)
        
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; background-color: {color}; margin: 20px 0;">
            <h3 style="color: white; margin: 0;">Risk Level: {risk_level}</h3>
            <p style="color: white; margin: 10px 0 0 0;">{message}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        recommendations = st.session_state.depression_predictor.get_recommendations(combined_score)
        for rec in recommendations:
            st.write(f"• {rec}")
        
        # Progress visualization
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=combined_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Mental Health Risk Assessment"},
            delta={'reference': 5.0},
            gauge={
                'axis': {'range': [None, 10]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 3], 'color': "lightgreen"},
                    {'range': [3, 5], 'color': "yellow"},
                    {'range': [5, 7], 'color': "orange"},
                    {'range': [7, 10], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 7
                }
            }
        ))
        
        st.plotly_chart(fig_gauge, use_container_width=True)

def show_resources_page():
    st.header("🆘 Mental Health Resources")
    
    resources = get_mental_health_resources()
    
    for category, items in resources.items():
        st.subheader(category)
        for item in items:
            st.write(f"• **{item['name']}**: {item['description']}")
            if 'contact' in item:
                st.write(f"  📞 {item['contact']}")
            if 'website' in item:
                st.write(f"  🌐 {item['website']}")
        st.markdown("---")

def show_about_page():
    st.header("ℹ️ About This Tool")
    
    st.markdown("""
    ### Purpose
    This Mental Health Assessment Tool uses natural language processing and basic audio analysis to provide insights into mental health indicators based on text and voice inputs.
    
    ### How It Works
    
    **Text Analysis:**
    - Analyzes emotional keywords and sentiment patterns
    - Identifies positive indicators (happiness, joy, optimism)
    - Detects concerning patterns (sadness, hopelessness, distress)
    - Uses advanced NLP to understand context and meaning
    
    **Voice Analysis:**
    - Examines speech energy levels and vocal patterns
    - Analyzes speaking pace and pause patterns
    - Detects emotional tone through acoustic features
    - Identifies indicators of mood through voice characteristics
    
    **Risk Scoring:**
    - **0-3 (Low Risk)**: Happy, positive content with good emotional indicators
    - **3-5 (Moderate Risk)**: Neutral content with mixed or unclear emotional signals
    - **5-7 (High Risk)**: Sad, concerning content with multiple risk indicators
    - **7-10 (Very High Risk)**: Severely distressed content requiring immediate attention
    
    ### Important Notes
    - This tool is for educational purposes only
    - Results should not replace professional medical advice
    - If you're experiencing mental health concerns, please seek professional help
    - All analysis is performed locally - your data is not stored or shared
    
    ### Technology
    - Natural Language Processing with NLTK and spaCy
    - Sentiment analysis using VADER and TextBlob
    - Audio processing with advanced pattern recognition
    - Machine learning-based risk assessment algorithms
    """)

def show_download_page():
    import zipfile
    import io
    from pathlib import Path
    
    st.header("📦 Download Project Files")
    
    st.markdown("""
    ### Get the Complete Mental Health Assessment Tool
    
    Download all the source code files to run this application on your own computer.
    """)
    
    def create_project_zip():
        """Create a zip file containing all project files"""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Files to include
            files_to_include = [
                'app.py',
                'text_analyzer.py', 
                'voice_analyzer_simple.py',
                'depression_predictor.py',
                'utils.py',
                'pyproject.toml'
            ]
            
            for filename in files_to_include:
                if os.path.exists(filename):
                    zip_file.write(filename, f'mental_health_app/{filename}')
            
            # Create .streamlit directory in zip
            config_content = """[server]
headless = true
address = "0.0.0.0"
port = 5000
"""
            zip_file.writestr('mental_health_app/.streamlit/config.toml', config_content)
            
            # Add requirements.txt
            requirements_content = """streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
nltk>=3.8.0
spacy>=3.6.0
textblob>=0.17.0
pydub>=0.25.0"""
            zip_file.writestr('mental_health_app/requirements.txt', requirements_content)
            
            # Add README.md
            readme_content = """# Mental Health Assessment Tool

A Streamlit-based web application that analyzes text and voice input to provide mental health risk assessments.

## Installation

1. Install Python 3.11 or higher
2. Install dependencies: `pip install -r requirements.txt`
3. Download spaCy model: `python -m spacy download en_core_web_sm`

## Usage

Run: `streamlit run app.py --server.port 5000`

## Risk Scoring

- 0-3: Low Risk (positive emotions)
- 3-5: Moderate Risk (neutral) 
- 5-7: High Risk (concerning)
- 7-10: Very High Risk (serious distress)

**Disclaimer:** For educational purposes only. Not a substitute for professional medical advice.
"""
            zip_file.writestr('mental_health_app/README.md', readme_content)
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("📥 Create Download", type="primary"):
            try:
                zip_data = create_project_zip()
                st.session_state.zip_ready = True
                st.session_state.zip_data = zip_data
                st.success("Download ready!")
            except Exception as e:
                st.error(f"Error creating download: {str(e)}")
    
    with col2:
        if hasattr(st.session_state, 'zip_ready') and st.session_state.zip_ready:
            st.download_button(
                label="💾 Download mental_health_app.zip",
                data=st.session_state.zip_data,
                file_name="mental_health_app.zip",
                mime="application/zip",
                type="primary"
            )
    
    st.markdown("""
    ### What's included:
    - `app.py` - Main application
    - `text_analyzer.py` - Text analysis
    - `voice_analyzer_simple.py` - Voice analysis  
    - `depression_predictor.py` - Risk prediction
    - `utils.py` - Resources and utilities
    - `requirements.txt` - Dependencies
    - `.streamlit/config.toml` - Configuration
    - `README.md` - Setup instructions
    
    ### Installation Instructions:

    1. **Extract the zip file** to your desired location
    2. **Open terminal/command prompt** in the extracted folder
    3. **Install dependencies:**
       ```bash
       pip install -r requirements.txt
       python -m spacy download en_core_web_sm
       ```
    4. **Run the application:**
       ```bash
       streamlit run app.py --server.port 5000
       ```
    5. **Open your browser** to `http://localhost:5000`
    """)

if __name__ == "__main__":
    main()
