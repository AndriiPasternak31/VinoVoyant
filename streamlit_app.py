import streamlit as st
import pandas as pd
import nltk
from src.models.country_predictor import WineCountryPredictor
from src.models.advanced_predictors import TransformerPredictor, DetailedPromptPredictor
import plotly.graph_objects as go
import os
import logging
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

nltk.download('punkt')
nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Page config
st.set_page_config(
    page_title="VinoVoyant - Wine Origin Predictor",
    page_icon="🍷",
    layout="wide"
)

# Add navigation
page = st.sidebar.radio("Navigation", ["Prediction", "Analytics"])

if page == "Prediction":
    # Initialize session state for API key
    if 'candidate_api_key' not in st.session_state:
        # Try to get API key from Streamlit secrets first
        try:
            st.session_state.candidate_api_key = st.secrets["candidate"]["api_key"]
            st.session_state.api_key_configured = True
        except:
            st.session_state.candidate_api_key = None
            st.session_state.api_key_configured = False

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

    # Initialize predictors
    if 'traditional_predictor' not in st.session_state:
        with st.spinner("Initializing traditional model..."):
            st.session_state.traditional_predictor = WineCountryPredictor()
            model_path = 'models/wine_country_predictor.joblib'
            
            # Force a data reload to see where it's coming from
            logger.info("Training new model to check data source...")
            try:
                metrics = st.session_state.traditional_predictor.train("data/wine_quality_1000.csv")
                st.session_state.traditional_predictor.save_model(model_path)
                logger.info("Successfully trained and saved new model")

                # Update data source indicator with more details
                data_source = getattr(st.session_state.traditional_predictor.preprocessor, 'last_data_source', 'Unknown')
                if data_source == "S3":
                    data_source_placeholder.success("✅ Data loaded from AWS S3")
                elif data_source == "Local File":
                    data_source_placeholder.warning("⚠️ Using local file (S3 access failed)")
                else:
                    data_source_placeholder.error("❌ Unknown data source")
                
                # Display model metrics in sidebar
                with st.sidebar:
                    st.header("📊 Model Performance")
                    
                    # Overall metrics
                    st.subheader("Overall Metrics")
                    overall_metrics = metrics['basic_metrics']['weighted avg']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Precision", f"{overall_metrics['precision']:.2f}")
                        st.metric("Recall", f"{overall_metrics['recall']:.2f}")
                    with col2:
                        st.metric("F1-Score", f"{overall_metrics['f1-score']:.2f}")
                        st.metric("Support", overall_metrics['support'])
                    
                    # Show confusion matrix
                    st.subheader("Confusion Matrix")
                    fig = go.Figure(data=go.Heatmap(
                        z=metrics['confusion_matrix']['matrix'],
                        x=metrics['confusion_matrix']['labels'],
                        y=metrics['confusion_matrix']['labels'],
                        colorscale='Blues'
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance
                    st.subheader("Top Features")
                    for feature, importance in metrics['feature_importance'].items():
                        st.metric(feature, f"{importance:.3f}")
                    
            except Exception as e:
                logger.error(f"Error during model training: {str(e)}")
                data_source_placeholder.error("❌ Error loading data")

    if 'transformer_predictor' not in st.session_state:
        with st.spinner("Initializing transformer model..."):
            st.session_state.transformer_predictor = TransformerPredictor(use_sagemaker=False)  # Use local mode for now

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
