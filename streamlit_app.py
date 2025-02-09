import streamlit as st
import pandas as pd
from src.models.country_predictor import WineCountryPredictor
from src.models.advanced_predictors import OpenAIEmbeddingPredictor, PromptEngineeringPredictor
import plotly.graph_objects as go
import os

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Page config
st.set_page_config(
    page_title="VinoVoyant - Wine Origin Predictor",
    page_icon="🍷",
    layout="wide"
)

# Initialize session state for API key
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = None
if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = False

# App title and description
st.title("🍷 VinoVoyant - Wine Origin Predictor")
st.markdown("""
Discover the likely origin of a wine based on its description! This AI-powered tool analyzes wine descriptions
to predict the country of origin using multiple prediction methods. Simply enter a wine description below to get started.
""")

# API Key Configuration Section
with st.sidebar:
    st.header("🔑 API Configuration")
    api_key = st.text_input(
        "Enter your OpenAI API Key",
        type="password",
        help="Get your API key from https://platform.openai.com/api-keys",
        key="api_key_input"
    )
    
    if st.button("Configure API Key"):
        if api_key:
            st.session_state.openai_api_key = api_key
            st.session_state.api_key_configured = True
            st.success("API Key configured successfully!")
        else:
            st.error("Please enter an API key")

# Initialize predictors based on API key status
if 'traditional_predictor' not in st.session_state:
    st.session_state.traditional_predictor = WineCountryPredictor()
    model_path = 'models/wine_country_predictor.joblib'
    
    if os.path.exists(model_path):
        try:
            st.session_state.traditional_predictor.load_model(model_path)
        except Exception as e:
            with st.spinner("Training traditional model..."):
                report = st.session_state.traditional_predictor.train("data/wine_quality_1000.csv")
                st.session_state.traditional_predictor.save_model(model_path)
    else:
        with st.spinner("Training traditional model..."):
            report = st.session_state.traditional_predictor.train("data/wine_quality_1000.csv")
            st.session_state.traditional_predictor.save_model(model_path)

# Initialize OpenAI predictors only if API key is configured
if st.session_state.api_key_configured:
    if 'openai_predictor' not in st.session_state:
        with st.spinner("Initializing OpenAI models..."):
            st.session_state.openai_predictor = OpenAIEmbeddingPredictor(st.session_state.openai_api_key)
            try:
                st.session_state.openai_predictor.load_model()
            except:
                report = st.session_state.openai_predictor.train("data/wine_quality_1000.csv")
                st.session_state.openai_predictor.save_model()
    
    if 'prompt_predictor' not in st.session_state:
        st.session_state.prompt_predictor = PromptEngineeringPredictor(st.session_state.openai_api_key)

# Main prediction section
st.header("🌍 Predict Wine Origin")

# Select prediction method based on API key status
available_methods = ["Traditional ML (TF-IDF + Logistic Regression)"]
if st.session_state.api_key_configured:
    available_methods.extend([
        "OpenAI Embeddings + ML",
        "GPT-4 Prompt Engineering"
    ])

prediction_method = st.radio(
    "Choose Prediction Method:",
    available_methods,
    help="Select the AI method to use for prediction. Advanced methods require OpenAI API key."
)

# Text input for wine description
wine_description = st.text_area(
    "Enter a wine description:",
    height=150,
    placeholder="Example: Very good Dry Creek Zin, robust and dry and spicy. Really gets the tastebuds watering, with its tart flavors of sour cherry candy, red currants, blueberries, tart raisins and oodles of peppery spices. Drink this lusty wine with barbecue."
)

if st.button("Predict Origin"):
    if wine_description:
        with st.spinner("Analyzing wine description..."):
            # Get prediction based on selected method
            if prediction_method == "Traditional ML (TF-IDF + Logistic Regression)":
                prediction = st.session_state.traditional_predictor.predict(wine_description)
                show_reasoning = False
            elif prediction_method == "OpenAI Embeddings + ML" and st.session_state.api_key_configured:
                prediction = st.session_state.openai_predictor.predict(wine_description)
                show_reasoning = False
            elif st.session_state.api_key_configured:  # GPT-4 Prompt Engineering
                prediction = st.session_state.prompt_predictor.predict(wine_description)
                show_reasoning = True
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Prediction Results")
                
                # Create a gauge chart for confidence
                top_prediction = prediction['top_3_predictions'][0]
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=top_prediction['probability'] * 100,
                    title={'text': f"Confidence for {top_prediction['country']}"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 33], 'color': "lightgray"},
                            {'range': [33, 66], 'color': "gray"},
                            {'range': [66, 100], 'color': "darkgray"}
                        ],
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Top 3 Predictions")
                for pred in prediction['top_3_predictions']:
                    st.metric(
                        label=pred['country'],
                        value=f"{pred['probability']*100:.1f}%"
                    )
            
            # Display analysis details
            st.subheader("Analysis Details")
            
            if show_reasoning:
                st.write("### Reasoning")
                st.write(prediction.get('reasoning', 'No reasoning provided'))
            else:
                st.write("Key terms that influenced the prediction:")
                if prediction_method == "Traditional ML (TF-IDF + Logistic Regression)":
                    processed_text = st.session_state.traditional_predictor.preprocessor.preprocess_text(wine_description)
                    st.code(processed_text)
                else:
                    st.write("(Advanced embedding analysis - individual terms not available)")
    else:
        st.error("Please enter a wine description first!")

# Method comparison
st.markdown("---")
st.markdown("""
### 🔍 Prediction Methods Explained:

1. **Traditional ML (TF-IDF + Logistic Regression)**
   - Uses traditional text processing and machine learning
   - Fast and lightweight
   - Available without API key

2. **OpenAI Embeddings + ML**
   - Uses advanced language model embeddings
   - Better understanding of context and nuance
   - Requires OpenAI API key

3. **GPT-4 Prompt Engineering**
   - Uses advanced language model reasoning
   - Provides detailed analysis and explanation
   - Requires OpenAI API key
""")

# Footer
st.markdown("---")
st.markdown("by Andrii Pasternak!")
