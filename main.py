import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentiment_analyzer import preprocess_text  # Import the function

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4') 

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv')
df = df.head(30)
# Define the preprocess_text function

# Apply the function to the 'reviewText' column
df['reviewText'] = df['reviewText'].apply(preprocess_text)

# Display the processed DataFrame
print(df.head())

# Main function (optional, if you want to structure your code this way)
def main():
    nltk.download('words')

    print("starting")
    # Reapply preprocessing (if needed) inside main
    df['reviewText'] = df['reviewText'].apply(preprocess_text)

    print(df.head())

if __name__ == "__main__":
    main()
