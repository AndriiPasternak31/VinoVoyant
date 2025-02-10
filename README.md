# 🍷 VinoVoyant - Wine Origin Predictor

An AI-powered Streamlit app that predicts a wine's country of origin based on its description!

## Try It Out!

Visit the live app: [VinoVoyant on Streamlit](https://vinovoyant.streamlit.app)

All features, including OpenAI-powered predictions, are available in the deployed version!

## Features

- Traditional ML prediction using TF-IDF and Logistic Regression
- Advanced prediction using OpenAI embeddings
- Expert analysis using GPT-4 prompt engineering
- Beautiful visualization of prediction confidence
- Detailed analysis of wine descriptions
- Data hosted on AWS S3 for reliable access and scalability

## Data Source

The wine dataset is hosted on AWS S3 at:
```
vino-voyant-wine-origin-predictor.s3.eu-north-1.amazonaws.com/wine_quality_1000.csv
```
The application automatically fetches the latest data from S3, with a fallback to local files if needed.

## Local Development Setup

If you want to run the app locally or contribute to development:

1. Clone the repository:
   ```bash
   git clone https://github.com/AndriiPasternak31/VinoVoyant.git
   cd VinoVoyant
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key:
   - Option 1: Enter your API key in the app's sidebar when running locally
   - Option 2: Create a `.env` file with your API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

5. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

## Project Structure

The app uses a modular structure:
- `src/preprocessing.py`: Data preprocessing utilities
- `src/models/`: Model implementations
- `streamlit_app.py`: Main application interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Development

- The app uses a modular structure with separate modules for:
  - Data preprocessing (`src/preprocessing.py`)
  - Model implementations (`src/models/`)
  - Configuration management (`src/config.py`)

## Deployment

When deploying to Streamlit Cloud:
1. Add your OpenAI API key to the app's secrets
2. No other configuration needed - just connect your GitHub repository!

## Security Note

Never commit sensitive information like API keys to the repository. Use environment variables or Streamlit's secrets management instead.
