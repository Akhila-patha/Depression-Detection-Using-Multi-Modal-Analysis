# Mental Health Assessment Tool

A Streamlit-based web application that analyzes text and voice input to provide mental health risk assessments.

## Installation

1. Install Python 3.11 or higher
2. Extract this zip file to your desired location
3. Open terminal/command prompt in the extracted folder
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

## Usage

Run the application:
```bash
streamlit run app.py --server.port 5000
```

Then open your browser to: http://localhost:5000

## Risk Scoring

- 0-3: Low Risk (positive emotions)
- 3-5: Moderate Risk (neutral) 
- 5-7: High Risk (concerning)
- 7-10: Very High Risk (serious distress)

## Features

- Text analysis using sentiment analysis and keyword detection
- Voice analysis processing audio patterns
- Combined risk assessment with recommendations
- Mental health resources and support information

**Important Disclaimer:** For educational purposes only. Not a substitute for professional medical advice.

## Files Included

- app.py - Main Streamlit application
- text_analyzer.py - Text analysis engine
- voice_analyzer_simple.py - Voice pattern analysis  
- depression_predictor.py - Risk prediction logic
- utils.py - Resources and utility functions
- requirements.txt - Python dependencies
- .streamlit/config.toml - Application configuration
