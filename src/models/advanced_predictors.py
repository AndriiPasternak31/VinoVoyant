import os
import requests
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Filter out PyTorch warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='torch')

# Lazy import of torch and transformers
def get_torch_modules():
    try:
        import torch
        from transformers import DistilBertTokenizer, DistilBertModel
        return torch, DistilBertTokenizer, DistilBertModel
    except ImportError as e:
        logger.error(f"Failed to import PyTorch modules: {str(e)}")
        return None, None, None

import boto3
import json
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

class TransformerPredictor:
    def __init__(self, use_sagemaker=False):
        self.use_sagemaker = use_sagemaker
        self.label_encoder = None
        self.model_initialized = False
        self.torch = None
        self.fallback_mode = False
        self.fallback_vectorizer = TfidfVectorizer(max_features=768)  # Same dim as DistilBERT
        self.fallback_classifier = LogisticRegression(max_iter=1000)
        
        if not use_sagemaker:
            try:
                # Lazy load PyTorch modules
                torch, DistilBertTokenizer, DistilBertModel = get_torch_modules()
                if None in (torch, DistilBertTokenizer, DistilBertModel):
                    logger.warning("Failed to load PyTorch modules, switching to fallback mode")
                    self.fallback_mode = True
                else:
                    # Store torch module
                    self.torch = torch
                    
                    # Local model initialization
                    self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                    self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
                    self.classifier = LogisticRegression(max_iter=1000)
                    self.model_initialized = True
                    logger.info("Successfully initialized local transformer model")
                
                # Train the model immediately
                try:
                    logger.info("Training classifier...")
                    self.train("data/wine_quality_1000.csv")
                    logger.info("Successfully trained classifier")
                    
                    # Save the trained model
                    self.save_model()
                except Exception as e:
                    logger.error(f"Error training classifier: {str(e)}")
            except Exception as e:
                logger.error(f"Error in initialization: {str(e)}")
                logger.warning("Switching to fallback mode")
                self.fallback_mode = True
        else:
            try:
                # SageMaker endpoint initialization
                self.endpoint_name = "vinovoyant-transformer"
                self.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
            except Exception as e:
                logger.error(f"Error initializing SageMaker client: {str(e)}")
                logger.warning("Switching to fallback mode")
                self.fallback_mode = True
    
    def get_embeddings(self, texts):
        """Get embeddings from either transformer model or fallback TF-IDF."""
        if not isinstance(texts, list):
            texts = [texts]
        
        if self.fallback_mode:
            try:
                # Use TF-IDF as fallback
                features = self.fallback_vectorizer.transform(texts)
                return features.toarray()
            except Exception as e:
                logger.error(f"Error in fallback embeddings: {str(e)}")
                raise
        
        if not self.use_sagemaker:
            try:
                # Ensure torch is available
                if self.torch is None:
                    torch, _, _ = get_torch_modules()
                    if torch is None:
                        raise ImportError("Failed to import PyTorch")
                    self.torch = torch
                
                # Local embedding generation
                inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
                with self.torch.no_grad():
                    outputs = self.model(**inputs)
                # Use [CLS] token embeddings
                embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                return embeddings
            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}")
                # Switch to fallback mode if transformer fails
                logger.warning("Switching to fallback mode")
                self.fallback_mode = True
                return self.get_embeddings(texts)
        else:
            # SageMaker inference
            try:
                payload = {"texts": texts}
                response = self.sagemaker_runtime.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    ContentType='application/json',
                    Body=json.dumps(payload)
                )
                embeddings = np.array(json.loads(response['Body'].read()))
                return embeddings
            except Exception as e:
                logger.error(f"SageMaker inference failed: {str(e)}")
                # Switch to fallback mode if SageMaker fails
                logger.warning("Switching to fallback mode")
                self.fallback_mode = True
                return self.get_embeddings(texts)

    def train(self, data_path):
        """Train the classifier using either transformer or fallback embeddings."""
        try:
            # Load and preprocess data
            df = pd.read_csv(data_path)
            df = df.dropna(subset=['description', 'country'])
            
            # Get embeddings
            if self.fallback_mode:
                logger.info("Training with fallback TF-IDF features")
                X = self.fallback_vectorizer.fit_transform(df['description']).toarray()
            else:
                X = self.get_embeddings(df['description'].tolist())
            
            # Prepare target variable
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(df['country'])
            
            # Split and train
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if self.fallback_mode:
                self.fallback_classifier.fit(X_train, y_train)
                y_pred = self.fallback_classifier.predict(X_test)
            elif not self.use_sagemaker:
                self.classifier.fit(X_train, y_train)
                y_pred = self.classifier.predict(X_test)
            else:
                # Use SageMaker endpoint for predictions
                embeddings = self.get_embeddings(df['description'].iloc[y_test.index].tolist())
                response = self.sagemaker_runtime.invoke_endpoint(
                    EndpointName=f"{self.endpoint_name}-classifier",
                    ContentType='application/json',
                    Body=json.dumps({"embeddings": embeddings.tolist()})
                )
                y_pred = np.array(json.loads(response['Body'].read()))
            
            # Return performance report
            from sklearn.metrics import classification_report
            return classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            raise

    def predict(self, description):
        """Predict country using either transformer or fallback model."""
        try:
            embeddings = self.get_embeddings(description)
            
            if self.fallback_mode:
                pred_encoded = self.fallback_classifier.predict(embeddings)
                pred_proba = self.fallback_classifier.predict_proba(embeddings)
            elif not self.use_sagemaker:
                pred_encoded = self.classifier.predict(embeddings)
                pred_proba = self.classifier.predict_proba(embeddings)
            else:
                # Use SageMaker endpoint for predictions
                response = self.sagemaker_runtime.invoke_endpoint(
                    EndpointName=f"{self.endpoint_name}-classifier",
                    ContentType='application/json',
                    Body=json.dumps({"embeddings": embeddings.tolist()})
                )
                result = json.loads(response['Body'].read())
                pred_encoded = np.array(result['predictions'])
                pred_proba = np.array(result['probabilities'])
            
            predicted_country = self.label_encoder.inverse_transform(pred_encoded)
            
            # Get top 3 predictions
            top_3_indices = np.argsort(pred_proba[0])[-3:][::-1]
            top_3_countries = self.label_encoder.inverse_transform(top_3_indices)
            top_3_probas = pred_proba[0][top_3_indices]
            
            return {
                'predicted_country': predicted_country[0],
                'top_3_predictions': [
                    {'country': country, 'probability': float(prob)}
                    for country, prob in zip(top_3_countries, top_3_probas)
                ]
            }
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return {
                'predicted_country': 'Error',
                'top_3_predictions': [
                    {'country': 'Error', 'probability': 0.0},
                    {'country': 'Error', 'probability': 0.0},
                    {'country': 'Error', 'probability': 0.0}
                ]
            }
    
    def save_model(self, model_path='models/wine_country_predictor_transformer.joblib'):
        """Save the trained model."""
        if not self.use_sagemaker:
            try:
                model_data = {
                    'classifier': self.classifier,
                    'label_encoder': self.label_encoder
                }
                joblib.dump(model_data, model_path)
            except Exception as e:
                logger.error(f"Error saving model: {str(e)}")
                raise
    
    def load_model(self, model_path='models/wine_country_predictor_transformer.joblib'):
        """Load a trained model."""
        if not self.use_sagemaker:
            try:
                model_data = joblib.load(model_path)
                self.classifier = model_data['classifier']
                self.label_encoder = model_data['label_encoder']
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise

class DetailedPromptPredictor:
    def __init__(self, api_key=None):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("Candidate API key is required")
        self.api_url = "https://candidate-llm.extraction.artificialos.com/v1/chat/completions"  # Fixed complete URL
        self.headers = {
            "x-api-key": self.api_key,  # Using x-api-key as specified
            "Content-Type": "application/json"
        }
        
    def create_analysis_prompt(self, description):
        return f"""Analyze this wine description and determine its likely country of origin between US, Spain, France, and Italy. 
        Consider these aspects:
        1. Wine style and characteristics
        2. Winemaking techniques mentioned
        3. Grape varieties if mentioned
        4. Climate indicators in the description
        5. Regional terminology used
        6. Typical wine styles from each country:
           - US: Bold, fruit-forward, high alcohol, modern techniques
           - Spain: Tempranillo, Garnacha, oak aging, traditional methods
           - France: Elegant, terroir-focused, strict regulations
           - Italy: Indigenous varieties, food-friendly, regional diversity

        Wine description: "{description}"

        Provide your analysis in this JSON format:
        {{
            "predicted_country": "country name (one of: US, Spain, France, Italy)",
            "confidence": "probability between 0 and 1",
            "alternative_countries": [
                {{"country": "second most likely", "probability": "probability"}},
                {{"country": "third most likely", "probability": "probability"}}
            ],
            "reasoning": "detailed explanation of why this wine matches the predicted country's style"
        }}
        """

    def predict(self, description):
        """Predict country using prompt engineering."""
        try:
            logger.info(f"Making API request to {self.api_url}")  # Debug log
            logger.info(f"Headers: {self.headers}")  # Debug log (without showing full API key)
            
            response = requests.post(
                self.api_url,  # Using complete URL
                headers=self.headers,
                json={
                    "model": "gpt-4o-mini",  # Using the recommended model
                    "messages": [
                        {"role": "system", "content": "You are a master sommelier with extensive knowledge of wines from all regions."},
                        {"role": "user", "content": self.create_analysis_prompt(description)}
                    ],
                    "temperature": 0.3
                }
            )
            
            logger.info(f"Response status: {response.status_code}")  # Debug log
            logger.info(f"Response text: {response.text}")  # Debug log
            
            if response.status_code != 200:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                raise Exception(f"API request failed: {response.text}")
            
            # Get the content from the response
            content = response.json()['choices'][0]['message']['content']
            logger.info(f"Raw LLM response: {content}")  # Debug log
            
            # Clean up the markdown formatting
            json_str = content.replace("```json", "").replace("```", "").strip()
            logger.info(f"Cleaned JSON string: {json_str}")  # Debug log
            
            parsed_result = json.loads(json_str)
            
            return {
                'predicted_country': parsed_result['predicted_country'],
                'top_3_predictions': [
                    {'country': parsed_result['predicted_country'], 
                     'probability': float(parsed_result['confidence'])},
                    {'country': parsed_result['alternative_countries'][0]['country'], 
                     'probability': float(parsed_result['alternative_countries'][0]['probability'])},
                    {'country': parsed_result['alternative_countries'][1]['country'], 
                     'probability': float(parsed_result['alternative_countries'][1]['probability'])}
                ],
                'reasoning': parsed_result['reasoning']
            }
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")  # Debug log
            return {
                'predicted_country': 'Error',
                'top_3_predictions': [
                    {'country': 'Error', 'probability': 0.0},
                    {'country': 'Error', 'probability': 0.0},
                    {'country': 'Error', 'probability': 0.0}
                ],
                'reasoning': f'Failed to parse prediction: {str(e)}'
            }

class CandidatePredictor:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('CANDIDATE_API_KEY')
        if not self.api_key:
            raise ValueError("Candidate API key is required")
        self.api_url = "https://candidate-llm.extraction.artificialos.com/v1"
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
    def create_analysis_prompt(self, description):
        return f"""Analyze this wine description and determine its likely country of origin. 
        Consider these aspects:
        1. Wine style and characteristics
        2. Winemaking techniques mentioned
        3. Grape varieties if mentioned
        4. Climate indicators in the description
        5. Regional terminology used

        Wine description: "{description}"

        Provide your analysis in this JSON format:
        {{
            "predicted_country": "country name",
            "confidence": "probability between 0 and 1",
            "alternative_countries": [
                {{"country": "second most likely", "probability": "probability"}},
                {{"country": "third most likely", "probability": "probability"}}
            ],
            "reasoning": "brief explanation of the prediction"
        }}
        """

    def predict(self, description):
        """Predict country using prompt engineering."""
        response = requests.post(
            f"{self.api_url}/chat/completions",
            headers=self.headers,
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a master sommelier with extensive knowledge of wines from all regions."},
                    {"role": "user", "content": self.create_analysis_prompt(description)}
                ],
                "temperature": 0.3
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.text}")
        
        try:
            import json
            result = json.loads(response.json()['choices'][0]['message']['content'])
            
            return {
                'predicted_country': result['predicted_country'],
                'top_3_predictions': [
                    {'country': result['predicted_country'], 'probability': float(result['confidence'])},
                    {'country': result['alternative_countries'][0]['country'], 
                     'probability': float(result['alternative_countries'][0]['probability'])},
                    {'country': result['alternative_countries'][1]['country'], 
                     'probability': float(result['alternative_countries'][1]['probability'])}
                ],
                'reasoning': result['reasoning']
            }
        except Exception as e:
            logger.error(f"Error parsing prediction: {str(e)}")
            return {
                'predicted_country': 'Error',
                'top_3_predictions': [
                    {'country': 'Error', 'probability': 0.0},
                    {'country': 'Error', 'probability': 0.0},
                    {'country': 'Error', 'probability': 0.0}
                ],
                'reasoning': 'Failed to parse prediction'
            }

class PromptEngineeringPredictor:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('CANDIDATE_API_KEY')
        if not self.api_key:
            raise ValueError("Candidate API key is required")
        self.api_url = "https://candidate-llm.extraction.artificialos.com/v1"
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
    def create_analysis_prompt(self, description):
        return f"""Analyze this wine description and determine its likely country of origin. 
        Consider these aspects:
        1. Wine style and characteristics
        2. Winemaking techniques mentioned
        3. Grape varieties if mentioned
        4. Climate indicators in the description
        5. Regional terminology used

        Wine description: "{description}"

        Provide your analysis in this JSON format:
        {{
            "predicted_country": "country name",
            "confidence": "probability between 0 and 1",
            "alternative_countries": [
                {{"country": "second most likely", "probability": "probability"}},
                {{"country": "third most likely", "probability": "probability"}}
            ],
            "reasoning": "brief explanation of the prediction"
        }}
        """

    def predict(self, description):
        """Predict country using prompt engineering."""
        response = requests.post(
            f"{self.api_url}/chat/completions",
            headers=self.headers,
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a master sommelier with extensive knowledge of wines from all regions."},
                    {"role": "user", "content": self.create_analysis_prompt(description)}
                ],
                "temperature": 0.3
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.text}")
        
        try:
            import json
            result = json.loads(response.json()['choices'][0]['message']['content'])
            
            return {
                'predicted_country': result['predicted_country'],
                'top_3_predictions': [
                    {'country': result['predicted_country'], 'probability': float(result['confidence'])},
                    {'country': result['alternative_countries'][0]['country'], 
                     'probability': float(result['alternative_countries'][0]['probability'])},
                    {'country': result['alternative_countries'][1]['country'], 
                     'probability': float(result['alternative_countries'][1]['probability'])}
                ],
                'reasoning': result['reasoning']
            }
        except Exception as e:
            logger.error(f"Error parsing prediction: {str(e)}")
            return {
                'predicted_country': 'Error',
                'top_3_predictions': [
                    {'country': 'Error', 'probability': 0.0},
                    {'country': 'Error', 'probability': 0.0},
                    {'country': 'Error', 'probability': 0.0}
                ],
                'reasoning': 'Failed to parse prediction'
            } 