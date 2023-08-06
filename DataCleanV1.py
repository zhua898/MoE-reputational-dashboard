import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string
import re


#ctrl + / = comment
#pandas default UTF-8 and comma as separator
df = pd.read_csv('20230724-Meltwater export.csv', encoding='UTF-16', sep='\t')
print(df.columns)
#print(df['Sentiment'].head(20))

#lowercase all content in the report
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.lower()
print(df.head(10))


#Column: URL
#check if URL is twitter/ non-twitter link
df['is_twitter'] = df['URL'].str.contains('twitter.com')
twitter_count = df['is_twitter'].sum()
non_twitter_count = len(df) - twitter_count
print(twitter_count)
print(non_twitter_count)


#Column: Influencer
#remove @ and replace NaN values with 'NULL'
df['Influencer'] = df['Influencer'].str.replace('@','')
df['Influencer'] = df['Influencer'].fillna('null')
print(df['Influencer'].head(10))


#Column: key phrases
#lowercase all key phrases and replace NaN values with 'NULL'
df['Key Phrases'] = df['Key Phrases'].str.lower()
df['Key Phrases'] = df['Key Phrases'].fillna('NULL')
#print some sample data to check if its replaced with 'NULL'
print(df['Key Phrases'].head(20))


#Column: Tweet Id & Twitter Id
#remove "" and keep only number; replace NaN values with 'NULL'
df['Tweet Id'] = df['Tweet Id'].str.replace('"', '')
df['Twitter Id'] = df['Twitter Id'].str.replace('"', '')
df['Tweet Id'] = df['Tweet Id'].fillna('NULL')
df['Twitter Id'] = df['Twitter Id'].fillna('NULL')
print(df['Tweet Id'].head(20))
print(df['Twitter Id'].head(20))
#count most appeared twitter ID/ tweet ID


#Column: URL & User Profile Url
#Remove https:// and replace NaN values with 'NULL'(non-tweets)
df['URL'] = df['URL'].str.replace('https://', '')
df['URL'] = df['URL'].str.replace('http://', '')
df['URL'] = df['URL'].fillna('NULL')

df['User Profile Url'] = df['User Profile Url'].str.replace('https://', '')
df['User Profile Url'] = df['User Profile Url'].str.replace('http://', '')
df['User Profile Url'] = df['User Profile Url'].fillna('NULL')
print(df['User Profile Url'].head(10))


#Sheffin
#column: Hit Sentence
#firstly replace NaN values with 'null'
df['Hit Sentence'] = df['Hit Sentence'].fillna('NULL')

# remove stop words, punctuation, and numbers or digits from the Hit sentence column
def process_text(text):
    #lowercase content
    text = text.lower()

    #remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    #remove digits
    text = re.sub(r'\d+', '', text)

    #remove stop words
    # tokens = text.split()
    # tokens = [token for token in tokens if token not in ENGLISH_STOP_WORDS]
    # text = ' '.join(tokens)

    #remove URLs
    text = re.sub(r'http\S+', '', text)

    #Remove Twitter mentions
    text = re.sub(r'@\w+', '', text)

    #Remove common words in Twitter (Example: "rt", "re", "amp" which refers to retweet, reply and "&")
    text = text.replace('rt', '') #retweets
    text = text.replace('amp', '') # &
    text = text.replace('re', '') #reply

    #remove additional special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    #remove specific common words
    # text = text.replace('nz','')

    #remove non-ASCII characters
    text = ''.join(character for character in text if ord(character) < 128)

    return text.strip()

#apply the defined process_text function to the column
df['Hit Sentence'] = df['Hit Sentence'].apply(process_text)
#print 10 sample data to check
print(df['Hit Sentence'].head(10))




#Sentiment rating from textblob
sentiments = []
for index, row in df.iterrows():
    text_to_analyze = row['Hit Sentence']
    if pd.notna(text_to_analyze):
        analysis = TextBlob(text_to_analyze)
        sentiment_polarity = analysis.sentiment.polarity

        # Classify the sentiment
        if sentiment_polarity < 0:
            sentiments.append(-1)
        elif sentiment_polarity == 0:
            sentiments.append(0)
        else:
            sentiments.append(1)

# Compute summary statistics for the sentiment polarities; use of numpy package
mean_sentiment = np.mean(sentiments)
median_sentiment = np.median(sentiments)
std_dev_sentiment = np.std(sentiments)

# Print the summary statistics
print("Mean Sentiment:", mean_sentiment)
print("Median Sentiment:", median_sentiment)
print("Standard Deviation of Sentiment:", std_dev_sentiment)

#visualize and plot the data (x-y axis, title, legend)
plt.hist(sentiments)
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.annotate(f'Mean: {mean_sentiment:.5f}', xy=(0.05, 0.85), xycoords='axes fraction')
plt.show()



#Summary statistics
#Twitter/Non-Twitter count
count_dict={
    'Count of Tweet links': twitter_count,
    'Count of non Tweet links': non_twitter_count,
    'Total count': twitter_count + non_twitter_count
}
summary_df = pd.DataFrame.from_dict(count_dict, orient='index')
summary_df.columns = ['Count']
print(summary_df)

#transform the summary stats to new CSV file
summary_df.to_csv('summary_stats.csv')




#because csv would change ID to scientific notation, the format is changed to xlsx for the output
df.to_excel('processed_meltwater_report.xlsx',index=False)
















