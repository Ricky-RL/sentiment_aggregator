import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentiment_analyzer import preprocess_text  # Import both functions

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv')
df = df.head(20)  # Keep only the first 10 lines

# Apply text preprocessing
df['reviewText'] = df['reviewText'].apply(preprocess_text)

# Display the processed DataFrame
print(df.head())

# Define the get_sentiment function
def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()  # Initialize the analyzer here
    scores = analyzer.polarity_scores(text)

    sentiment = 1 if scores['pos'] > scores['neg'] else 0  # Basic sentiment logic: positive or negative

    return sentiment

# Main function
def main():
    print("Starting sentiment analysis...")

    df['reviewText'] = df['reviewText'].apply(preprocess_text)  # Preprocess text again if necessary
    df['sentiment'] = df['reviewText'].apply(get_sentiment)  # Apply sentiment analysis

    print(df)

if __name__ == "__main__":
    main()
