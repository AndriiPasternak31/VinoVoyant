import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

class WineDataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(max_features=100)
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            print("Warning: NLTK data download failed. Some text processing features might be limited.")
    
    def load_data(self, file_path):
        """Load the wine dataset."""
        return pd.read_csv(file_path)
    
    def clean_data(self, df):
        """Basic data cleaning operations."""
        df_clean = df.copy()
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Handle missing values
        df_clean['price'] = df_clean['price'].fillna(df_clean['price'].median())
        df_clean['description'] = df_clean['description'].fillna('')
        df_clean['variety'] = df_clean['variety'].fillna('Unknown')
        df_clean['country'] = df_clean['country'].fillna('Unknown')
        
        return df_clean
    
    def preprocess_text(self, text):
        """Clean and preprocess text data."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        return ' '.join(tokens)
    
    def engineer_features(self, df):
        """Create new features from existing data."""
        df_engineered = df.copy()
        
        # Extract text length
        df_engineered['description_length'] = df_engineered['description'].str.len()
        
        # Create price bands
        df_engineered['price_category'] = pd.qcut(df_engineered['price'], 
                                                q=5, 
                                                labels=['very_cheap', 'cheap', 'medium', 'expensive', 'very_expensive'])
        
        # Create quality categories based on points
        df_engineered['quality_category'] = pd.qcut(df_engineered['points'],
                                                  q=3,
                                                  labels=['low', 'medium', 'high'])
        
        return df_engineered
    
    def encode_categorical(self, df, columns_to_encode=None):
        """Encode categorical variables."""
        if columns_to_encode is None:
            columns_to_encode = ['country', 'variety']
            
        df_encoded = df.copy()
        
        for col in columns_to_encode:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le
        
        return df_encoded
    
    def scale_numerical(self, df, columns_to_scale=None):
        """Scale numerical features."""
        if columns_to_scale is None:
            columns_to_scale = ['price', 'points']
            
        df_scaled = df.copy()
        
        scaling_data = df_scaled[columns_to_scale]
        scaled_values = self.scaler.fit_transform(scaling_data)
        
        for i, col in enumerate(columns_to_scale):
            df_scaled[f'{col}_scaled'] = scaled_values[:, i]
            
        return df_scaled
    
    def create_text_features(self, df, text_column='description'):
        """Create TF-IDF features from text data."""
        # Preprocess descriptions
        processed_descriptions = df[text_column].apply(self.preprocess_text)
        
        # Create TF-IDF features
        tfidf_features = self.tfidf.fit_transform(processed_descriptions)
        
        # Convert to DataFrame
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        return pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)
    
    def prepare_data(self, file_path, include_text_features=True):
        """Complete data preparation pipeline."""
        # Load data
        df = self.load_data(file_path)
        
        # Clean data
        df = self.clean_data(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Encode categorical variables
        df = self.encode_categorical(df)
        
        # Scale numerical features
        df = self.scale_numerical(df)
        
        # Create text features if requested
        if include_text_features:
            df = self.create_text_features(df)
        
        return df

# Example usage
if __name__ == "__main__":
    preprocessor = WineDataPreprocessor()
    processed_data = preprocessor.prepare_data("data/wine_quality_1000.csv")
    print("Processed data shape:", processed_data.shape)
    print("\nFeature columns:", processed_data.columns.tolist()) 