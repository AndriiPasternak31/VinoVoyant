# 🍷 VinoVoyant - Wine Origin Predictor

An AI-powered Streamlit app that predicts a wine's country of origin based on its description!

## Features

- Traditional ML prediction using TF-IDF and Logistic Regression
- Advanced prediction using OpenAI embeddings
- Expert analysis using GPT-4 prompt engineering
- Beautiful visualization of prediction confidence
- Detailed analysis of wine descriptions

## Setup Instructions

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

4. Set up environment variables:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your OpenAI API key
   - Or use the in-app API key configuration in the sidebar

5. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

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
