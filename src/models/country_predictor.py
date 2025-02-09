import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from src.preprocessing import WineDataPreprocessor

class WineCountryPredictor:
    def __init__(self):
        self.preprocessor = WineDataPreprocessor()
        self.model = LogisticRegression(max_iter=1000, multi_class='multinomial')
        self.tfidf = None
        self.label_encoder = None
        
    def prepare_features(self, descriptions):
        """Prepare text features for prediction."""
        if isinstance(descriptions, str):
            descriptions = [descriptions]
        
        # Preprocess descriptions
        processed_descriptions = [self.preprocessor.preprocess_text(desc) for desc in descriptions]
        
        # Transform using fitted TF-IDF
        if self.tfidf is not None:
            features = self.tfidf.transform(processed_descriptions)
            return features
        else:
            raise ValueError("Model not trained yet. Please train the model first.")
    
    def train(self, data_path):
        """Train the country prediction model."""
        # Load and preprocess data
        df = self.preprocessor.load_data(data_path)
        df = self.preprocessor.clean_data(df)
        
        # Process descriptions and create TF-IDF features
        processed_descriptions = df['description'].apply(self.preprocessor.preprocess_text)
        self.tfidf = self.preprocessor.tfidf
        X = self.tfidf.fit_transform(processed_descriptions)
        
        # Prepare target variable
        self.label_encoder = self.preprocessor.label_encoders.get('country', None)
        if self.label_encoder is None:
            df = self.preprocessor.encode_categorical(df, ['country'])
            self.label_encoder = self.preprocessor.label_encoders['country']
        
        y = self.label_encoder.transform(df['country'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)
        
        return report
    
    def predict(self, description):
        """Predict country for a given wine description."""
        # Prepare features
        features = self.prepare_features(description)
        
        # Make prediction
        pred_encoded = self.model.predict(features)
        pred_proba = self.model.predict_proba(features)
        
        # Convert prediction to country name
        predicted_country = self.label_encoder.inverse_transform(pred_encoded)
        
        # Get top 3 predictions with probabilities
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
    
    def save_model(self, model_path='models/wine_country_predictor.joblib'):
        """Save the trained model and associated transformers."""
        model_data = {
            'model': self.model,
            'tfidf': self.tfidf,
            'label_encoder': self.label_encoder
        }
        joblib.dump(model_data, model_path)
    
    def load_model(self, model_path='models/wine_country_predictor.joblib'):
        """Load a trained model and associated transformers."""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.tfidf = model_data['tfidf']
        self.label_encoder = model_data['label_encoder']

# Train and save model if running as main
if __name__ == "__main__":
    predictor = WineCountryPredictor()
    report = predictor.train("data/wine_quality_1000.csv")
    print("Model Performance:\n", report)
    predictor.save_model() 