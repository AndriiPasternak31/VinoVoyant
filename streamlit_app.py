import streamlit as st
import pandas as pd
import nltk
from src.models.country_predictor import WineCountryPredictor
from src.models.advanced_predictors import TransformerPredictor, DetailedPromptPredictor
import plotly.graph_objects as go
import os
import logging
import sys
from src.analytics import (
    load_and_preprocess_data,
    create_ratings_distribution,
    create_price_vs_points,
    create_avg_price_by_country,
    create_top_varieties,
    create_sentiment_analysis,
    create_correlation_heatmap,
    create_variety_price_distribution
)

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Configure environment variables
if not os.environ.get('STREAMLIT_DISABLE_TORCH_WATCH'):
    os.environ['STREAMLIT_DISABLE_TORCH_WATCH'] = 'true'

# Initialize NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        logger.warning(f"Failed to download NLTK data: {e}")

# Page config
st.set_page_config(
    page_title="VinoVoyant - Wine Origin Predictor",
    page_icon="🍷",
    layout="wide"
)

# Initialize predictors in session state
@st.cache_resource
def init_traditional_predictor():
    try:
        predictor = WineCountryPredictor()
        model_path = 'models/wine_country_predictor.joblib'
        if os.path.exists(model_path):
            predictor.load_model(model_path)
        else:
            predictor.train("data/wine_quality_1000.csv")
            predictor.save_model(model_path)
        return predictor
    except Exception as e:
        logger.error(f"Error initializing traditional predictor: {e}")
        return None

@st.cache_resource
def init_transformer_predictor():
    try:
        return TransformerPredictor(use_sagemaker=False)
    except Exception as e:
        logger.error(f"Error initializing transformer predictor: {e}")
        return None

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.candidate_api_key = None
    st.session_state.api_key_configured = False
    try:
        st.session_state.candidate_api_key = st.secrets["candidate"]["api_key"]
        st.session_state.api_key_configured = True
    except:
        pass

# Add navigation
page = st.sidebar.radio("Navigation", ["Prediction", "Analytics"])

if page == "Prediction":
    # Initialize models if not done yet
    if not st.session_state.initialized:
        with st.spinner("Initializing models..."):
            st.session_state.traditional_predictor = init_traditional_predictor()
            st.session_state.transformer_predictor = init_transformer_predictor()
            st.session_state.initialized = True

    # App title and description
    st.title("🍷 VinoVoyant - Wine Origin Predictor")
    st.markdown("""
    Discover the likely origin of a wine based on its description! This AI-powered tool analyzes wine descriptions
    to predict the country of origin using multiple prediction methods. Simply enter a wine description below to get started.
    """)

    # Add a data source indicator
    with st.sidebar:
        st.header("📊 Data Source")
        data_source_placeholder = st.empty()

    # Only show API Key input if not configured in secrets
    if not st.session_state.api_key_configured:
        with st.sidebar:
            st.header("🔑 API Configuration")
            api_key = st.text_input(
                "Enter your Candidate API Key",
                type="password",
                help="Use the provided candidate API key",
                key="api_key_input"
            )
            
            if st.button("Configure API Key"):
                if api_key:
                    st.session_state.candidate_api_key = api_key
                    st.session_state.api_key_configured = True
                    st.success("API Key configured successfully!")
                else:
                    st.error("Please enter an API key")

    # Initialize LLM predictor only if API key is configured
    if st.session_state.api_key_configured:
        if 'llm_predictor' not in st.session_state:
            st.session_state.llm_predictor = DetailedPromptPredictor(st.session_state.candidate_api_key)

    # Main prediction section
    st.header("🌍 Predict Wine Origin")

    # Select prediction method based on API key status
    available_methods = [
        "Traditional ML (TF-IDF + Logistic Regression)",
        "Transformer (DistilBERT)"
    ]
    if st.session_state.api_key_configured:
        available_methods.append("Expert LLM Analysis")

    prediction_method = st.radio(
        "Choose Prediction Method:",
        available_methods,
        help="Select the AI method to use for prediction."
    )

    # Text input for wine description
    wine_description = st.text_area(
        "Enter a wine description:",
        height=150,
        placeholder="Example: Very good Dry Creek Zin, robust and dry and spicy. Really gets the tastebuds watering, with its tart flavors of sour cherry candy, red currants, blueberries, tart raisins and oodles of peppery spices. Drink this lusty wine with barbecue."
    )

    def make_prediction(prediction_method, wine_description):
        """Make a prediction using the selected method."""
        if prediction_method == "Traditional ML (TF-IDF + Logistic Regression)":
            prediction = st.session_state.traditional_predictor.predict(wine_description)
            return prediction, False
        elif prediction_method == "Transformer (DistilBERT)":
            try:
                prediction = st.session_state.transformer_predictor.predict(wine_description)
                if prediction['predicted_country'] == 'Error':
                    st.error("Error making transformer prediction. Please try the traditional ML method instead.")
                    return None, False
                return prediction, False
            except Exception as e:
                st.error(f"Error with transformer prediction: {str(e)}")
                return None, False
        elif st.session_state.api_key_configured:  # Expert LLM Analysis
            try:
                prediction = st.session_state.llm_predictor.predict(wine_description)
                if prediction['predicted_country'] == 'Error':
                    st.error("Error making LLM prediction. The API might be unavailable. Please try another method.")
                    return None, True
                return prediction, True
            except Exception as e:
                st.error(f"Error with LLM prediction: {str(e)}")
                return None, True
        return None, False

    def display_prediction(prediction, show_reasoning):
        """Display the prediction results."""
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
                st.write("(Advanced analysis - individual terms not available)")

    if st.button("Predict Origin"):
        if wine_description:
            with st.spinner("Analyzing wine description..."):
                # Get prediction based on selected method
                prediction, show_reasoning = make_prediction(prediction_method, wine_description)
                
                if prediction is not None:
                    display_prediction(prediction, show_reasoning)
        else:
            st.error("Please enter a wine description first!")

    # Method comparison
    st.markdown("---")
    st.markdown("""
    ### 🔍 Prediction Methods Explained:

    1. **Traditional ML (TF-IDF + Logistic Regression)**
       - Uses traditional text processing and machine learning
       - Fast and lightweight

    2. **Transformer (DistilBERT)**
       - Uses state-of-the-art transformer model
       - Better understanding of language context

    3. **Expert LLM Analysis**
       - Uses GPT-4o-mini with wine expertise prompt
       - Provides detailed reasoning for predictions
       - Requires Candidate API key
    """)

elif page == "Analytics":
    st.title("📊 Wine Market Analytics")
    st.markdown("""
    Explore insights about wine characteristics, pricing, and market positioning across different countries.
    These visualizations help understand market trends and consumer preferences.
    """)
    
    # Load data for analytics
    with st.spinner("Loading data for analysis..."):
        df = load_and_preprocess_data()
        
    if df is not None:
        # Create tabs for different categories of visualizations
        tab1, tab2, tab3 = st.tabs(["Quality & Pricing", "Market Analysis", "Sentiment"])
        
        with tab1:
            st.subheader("Quality and Pricing Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                fig, insight = create_ratings_distribution(df)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("""
                **AI-Generated Business Insight:**
                """)
                st.info(insight)
            
            with col2:
                fig, insight = create_price_vs_points(df)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("""
                **AI-Generated Business Insight:**
                """)
                st.info(insight)
            
            fig, insight = create_correlation_heatmap(df)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **AI-Generated Business Insight:**
            """)
            st.info(insight)
        
        with tab2:
            st.subheader("Market Positioning Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                fig, insight = create_avg_price_by_country(df)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("""
                **AI-Generated Business Insight:**
                """)
                st.info(insight)
            
            with col2:
                fig, insight = create_variety_price_distribution(df)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("""
                **AI-Generated Business Insight:**
                """)
                st.info(insight)
            
            fig, insight = create_top_varieties(df)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **AI-Generated Business Insight:**
            """)
            st.info(insight)
        
        with tab3:
            st.subheader("Sentiment Analysis")
            fig, insight = create_sentiment_analysis(df)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **AI-Generated Business Insight:**
            """)
            st.info(insight)
    else:
        st.error("Unable to load data for analytics. Please check the data source.")

# Footer
st.markdown("---")
st.markdown("by Andrii Pasternak!")
