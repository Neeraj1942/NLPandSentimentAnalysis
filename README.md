
# Stock Market Prediction Project Using NLP And Sentiment Analysis


## Investment Portfolio Analysis with Natural Language Processing

This Python script integrates financial data analysis with sentiment analysis using NLP techniques. Here's an overview of what the script does:

1. **Package Installation**: Checks for required packages (`nltk`, `textblob`, `spacy`, `pydantic`, `backtrader`) and installs them if not found.

2. **Data Loading and Preprocessing**:
   - Loads stock price data from a CSV file (`Step2.2_ReturnData.csv`).
   - Loads and preprocesses news data from a ZIP file (`Raw Headline Data.zip`), extracting headlines and associated tickers.

3. **Combining Data**:
   - Merges stock return data with news data based on dates and tickers (`Step3_NewsAndReturnData.csv`).

4. **Sentiment Analysis**:
   - Computes sentiment scores using TextBlob for news headlines.
   - Trains multiple classification models (Logistic Regression, KNN, Decision Tree, SVM, Neural Networks, Random Forest) on labeled news data (`LabelledNewsData.csv`) for sentiment prediction.

5. **Visualization**:
   - Plots sentiment correlation with stock returns using scatter plots.

6. **LSTM Model Creation**:
   - Creates an LSTM model for sentiment prediction based on news headlines.
   - Trains the LSTM model and evaluates its accuracy.

7. **Financial Lexicon-based Sentiment Analysis**:
   - Uses a financial lexicon and VADER sentiment analysis to derive sentiment scores for news headlines.

8. **Backtrader Strategy Implementation**:
   - Defines a sentiment-based trading strategy (`SentimentStrat`) using the Backtrader library.
   - Runs the strategy for specified tickers over different time periods (`2012-01-01` to `2014-12-31` and `2016-01-01` to `2018-12-31`).
   - Evaluates and plots the strategy's performance, including portfolio value, profit, and trade execution details.

This script provides a comprehensive analysis pipeline combining financial data, NLP-based sentiment analysis, machine learning for sentiment prediction, and backtesting of sentiment-driven trading strategies.

## Usage

To use the script:

1. Ensure Python 3.x is installed on your system.
2. Install required libraries using `pip install -r requirements.txt`.
3. Run the script `portfolio_analysis.py`.
4. View the console output and generated plots for analysis results.

This script is designed for analyzing predefined datasets and conducting sentiment-driven trading strategy backtesting.

## Requirements

- Python 3.x
- Libraries used in include numpy, pandas, matplotlib, seaborn, scikit-learn, nltk, textblob, spacy, vaderSentiment, tensorflow, pydantic and backtrader.

## Author

Neeraj Dhulipala
