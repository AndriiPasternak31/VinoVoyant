import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import boto3
import botocore
import io
import logging

nltk.download('punkt')
nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data with better error handling
def ensure_nltk_data():
    """Ensure all required NLTK data is downloaded."""
    required_resources = {
        'tokenizers': ['punkt'],
        'corpora': ['stopwords']
    }
    
    for resource_type, resources in required_resources.items():
        for resource in resources:
            try:
                nltk.data.find(f'{resource_type}/{resource}')
                logger.info(f"Found NLTK resource: {resource}")
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                    logger.info(f"Successfully downloaded NLTK resource: {resource}")
                except Exception as e:
                    logger.warning(f"Failed to download NLTK resource {resource}: {str(e)}")

# Initialize NLTK resources
ensure_nltk_data()

class WineDataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(max_features=100)
        self.s3_bucket = "vino-voyant-wine-origin-predictor"
        self.s3_region = "eu-north-1"
        self.last_data_source = None  # Track the last successful data source
        
        # Configure AWS for public access
        boto3.setup_default_session(region_name=self.s3_region)
        logger.info(f"Initialized WineDataPreprocessor with S3 bucket: {self.s3_bucket} in region: {self.s3_region}")
        
        # Initialize stopwords with fallback
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.warning(f"Failed to load stopwords, using empty set: {str(e)}")
            self.stop_words = set()
    
    def load_data(self, file_path):
        """Load the wine dataset from S3."""
        logger.info("Attempting to load data from S3...")
        
        try:
            # Check AWS credentials
            session = boto3.Session()
            credentials = session.get_credentials()
            if credentials is None:
                logger.warning("No AWS credentials found - attempting anonymous access")
            else:
                logger.info("AWS credentials found")
            
            # Try anonymous access for public bucket
            s3 = boto3.client(
                's3',
                region_name=self.s3_region,
                aws_access_key_id='',
                aws_secret_access_key='',
                config=botocore.config.Config(signature_version=botocore.UNSIGNED)
            )
            logger.info(f"Initialized S3 client in region: {self.s3_region} with anonymous access")
            
            logger.info(f"Attempting to get object from bucket: {self.s3_bucket}")
            obj = s3.get_object(Bucket=self.s3_bucket, Key='wine_quality_1000.csv')
            logger.info("Successfully retrieved data from S3")
            df = pd.read_csv(io.BytesIO(obj['Body'].read()))
            logger.info(f"Successfully loaded DataFrame from S3 with shape: {df.shape}")
            self.last_data_source = "S3"
            return df
                
        except s3.exceptions.NoSuchBucket:
            logger.error(f"Bucket {self.s3_bucket} does not exist")
            raise Exception(f"S3 bucket '{self.s3_bucket}' does not exist")
        except s3.exceptions.NoSuchKey:
            logger.error("File wine_quality_1000.csv not found in bucket")
            raise Exception(f"File 'wine_quality_1000.csv' not found in S3 bucket '{self.s3_bucket}'")
        except Exception as e:
            logger.error(f"Failed to load data from S3: {str(e)}")
            raise Exception("Failed to load data from S3. Please ensure the S3 bucket and file are accessible.")
    
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
        """Clean and preprocess text data with fallback options."""
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning (always performed)
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        try:
            # Try NLTK tokenization
            tokens = word_tokenize(text)
        except Exception as e:
            logger.warning(f"NLTK tokenization failed, falling back to simple split: {str(e)}")
            tokens = text.split()
        
        # Remove stopwords if available
        tokens = [token for token in tokens if token not in self.stop_words]
        
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