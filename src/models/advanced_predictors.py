import os
import openai
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

class OpenAIEmbeddingPredictor:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        openai.api_key = self.api_key
        self.model = LogisticRegression(max_iter=1000)
        self.label_encoder = None
        
    def get_embedding(self, text):
        """Get embeddings from OpenAI API."""
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        # The new API returns the embedding directly in the data attribute
        return response.data[0].embedding
    
    def prepare_features(self, descriptions):
        """Convert descriptions to embeddings."""
        if isinstance(descriptions, str):
            descriptions = [descriptions]
        return np.array([self.get_embedding(desc) for desc in descriptions])
    
    def train(self, data_path):
        """Train the model using OpenAI embeddings."""
        # Load data
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['description', 'country'])
        
        # Get embeddings for all descriptions
        X = self.prepare_features(df['description'].tolist())
        
        # Prepare target variable
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['country'])
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Return performance report
        from sklearn.metrics import classification_report
        y_pred = self.model.predict(X_test)
        return classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)
    
    def predict(self, description):
        """Predict country using OpenAI embeddings."""
        features = self.prepare_features(description)
        pred_encoded = self.model.predict(features)
        pred_proba = self.model.predict_proba(features)
        
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
    
    def save_model(self, model_path='models/wine_country_predictor_openai.joblib'):
        """Save the trained model."""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder
        }
        joblib.dump(model_data, model_path)
    
    def load_model(self, model_path='models/wine_country_predictor_openai.joblib'):
        """Load a trained model."""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']

class PromptEngineeringPredictor:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        openai.api_key = self.api_key
        
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
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a master sommelier with extensive knowledge of wines from all regions."},
                {"role": "user", "content": self.create_analysis_prompt(description)}
            ],
            temperature=0.3
        )
        
        try:
            import json
            # Access the message content using the new API format
            result = json.loads(response.choices[0].message.content)
            
            # Format the response to match other predictors
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
            print(f"Error parsing prediction: {str(e)}")
            return {
                'predicted_country': 'Error',
                'top_3_predictions': [
                    {'country': 'Error', 'probability': 0.0},
                    {'country': 'Error', 'probability': 0.0},
                    {'country': 'Error', 'probability': 0.0}
                ],
                'reasoning': 'Failed to parse prediction'
            } 