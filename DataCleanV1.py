import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string
import re
from nltk.stem import PorterStemmer
from vader_sentiment.vader_sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from collections import Counter
import openpyxl
import nltk
from gensim import corpora
from gensim.models import LdaModel, Phrases
from nltk.stem import  WordNetLemmatizer
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import requests
from bs4 import BeautifulSoup
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from langdetect import detect, lang_detect_exception
from langdetect.lang_detect_exception import LangDetectException
from gensim.models import CoherenceModel
from concurrent.futures import ThreadPoolExecutor

#ctrl + / = comment
#pandas default UTF-8 and comma as separator
df = pd.read_csv('year_data.csv', encoding='UTF-16', sep='\t')
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
# df['URL'] = df['URL'].str.replace('https://', '')
# df['URL'] = df['URL'].str.replace('http://', '')
df['URL'] = df['URL'].fillna('NULL')

# df['User Profile Url'] = df['User Profile Url'].str.replace('https://', '')
# df['User Profile Url'] = df['User Profile Url'].str.replace('http://', '')
df['User Profile Url'] = df['User Profile Url'].fillna('NULL')
print(df['User Profile Url'].head(10))




#Sheffin
#column: Hit Sentence
#firstly replace NaN values with 'null'
df['Hit Sentence'] = df['Hit Sentence'].fillna('NULL')

#phrasal verb
ps = PorterStemmer()
phrasal_verb_dict = {
    'add up': 'calculate',
    'break out of': 'abandon',
    'bear on': 'influence',
    'broke down': 'collapse',
    'buy out': 'purchase',
    'buy up': 'purchase',
    'call for': 'require'
}

# remove stop words, punctuation, and numbers or digits from the Hit sentence column
def process_text(text):
    #replace phrasal verbs
    #for phrasal, replacement in phrasal_verb_dict.items():
    #    text = text.replace(phrasal, replacement)

    #remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    #remove digits
    text = re.sub(r'\d+', '', text)

    #remove URLs
    text = re.sub(r'http\S+', '', text)

    #Remove Twitter mentions
    text = re.sub(r'@\w+', '', text)

    #stem words
    #text = ' '.join([ps.stem(word) for word in text.split()])

    #remove stopwords (HUGE IMPACT ON SENTIMENT RATING)
    #stop_words = set(stopwords.words('english'))
    #text = ' '.join([word for word in text.split() if word not in stop_words])

    #Remove common words in Twitter (Example: "rt", "re", "amp" which refers to retweet, reply and "&") !! (HUGE IMPACT ON SENTIMENT RATING)
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


#Column: Sentiment
# replace neutral positive negative with 0 1 -1
def map_sentiment(sentiment):
    if sentiment == 'neutral':
        return 0
    if sentiment == 'not rated':
        return 0
    if sentiment == 'positive':
        return 1
    if sentiment == 'negative':
        return -1
    else:
        return None
df['Meltwater_sentiment'] = df['Sentiment'].apply(map_sentiment)



#TEXTBLOB sentiment rating
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



#VADER: Valence Aware Dictionary and sentiment Reasoner
#Tolerance is 0.05 under/above which it is classified as negative/positive
analyzer = SentimentIntensityAnalyzer()
def vader_analysis(text):
    va = analyzer.polarity_scores(text)
    #positive sentiment
    if va['compound'] >= 0.05:
        return 1
    #negative sentiment
    elif va['compound'] <= -0.05:
        return -1
    #neutral sentiment
    else:
        return 0

df['Vader_Sentiment'] = df['Hit Sentence'].apply(vader_analysis)
#print(df[['Hit Sentence', 'Vader_Sentiment']].head(10))

#get count for each sentiment
sentiment_counts = df['Vader_Sentiment'].value_counts().sort_index()

# Plot the distribution of VADER sentiment values
plt.figure(figsize=(10,6))
bars = plt.bar(sentiment_counts.index, sentiment_counts.values, color=['red', 'gray', 'green'])

# Add title and labels
plt.title('Distribution of VADER Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Number of Records')
plt.xticks(ticks=[-1, 0, 1], labels=['Negative', 'Neutral', 'Positive'], rotation=0)
plt.tight_layout()

# Add counts on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 20, round(yval, 2), ha='center', va='bottom')
plt.show()



#8/13
#word cloud
#combine all text in hit sentence into one single string
concat_text = " ".join(sentence for sentence in df['Hit Sentence'] if sentence != 'NULL')
wordcloud = WordCloud(background_color="white", max_words=100, contour_width=3, contour_color='steelblue').generate(concat_text)
plt.figure(figsize=(10,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Most Used Words/Topics in Hit Sentence")
plt.show()

#generate a new column which list the most mentioned words and its count
def tokenize(sentence):
    words = re.findall(r'\b\w+\b', sentence)
    return words

# Combine all 'Hit Sentence' into one list
all_words = [word for sentence in df['Hit Sentence'] for word in tokenize(sentence)]
# Count word occurrence using the Counter method
word_counts = Counter(all_words)
# Get most common words and rank them
most_common_words = word_counts.most_common(100)
#use loc to make sure the column align correct
words, counts = zip(*most_common_words)
df.loc[:len(words)-1, 'Most Common Words'] = words
df.loc[:len(counts)-1, 'Count for most common words'] = counts




#8/18
#LDA
nltk.download('stopwords')
nltk.download('wordnet')


df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y %I:%M%p')
df['Month-Year'] = df['Date'].dt.to_period('M')

lemmatizer = WordNetLemmatizer()
#exclude useless words
excluded_words = {'stated', 'going', 'null', "said", "would", "also", "one", "education", "school", "children",
                  "ministry", "sector", "teacher", "teachers", "government", "schools", "kids", "home", "students",
                  "classes", "parents", "child", "staff", "families", "person", "percent", "work", "rain",
                  "year", "since", "last", "group", "whether", "asked", "new", "zealand", "say", "search",
                  "people", "way", "time", "point", "thing", "part", "something", "student", "te", "name", "m", "use",
                  "say", "made", "month", "day", "moe"
            }

stop_words = set(stopwords.words('english')).union(excluded_words)

for month_year, group in df.groupby('Month-Year'):
    #tokenize, remove stopwords, lemmatize and filter non-alpha tokens
    sentences = [nltk.word_tokenize(sent.lower()) for sent in group['Hit Sentence']]
    cleaned_sentences = [
        [lemmatizer.lemmatize(token) for token in sentence if token not in stop_words and token.isalpha() and len(token) > 2]
        for sentence in sentences
    ]

    #list possible combination of 2/3 common words
    bigram_model = Phrases(cleaned_sentences, min_count=5, threshold=100)
    trigram_model = Phrases(bigram_model[cleaned_sentences], threshold=100)
    tokens_with_bigrams = [bigram_model[sent] for sent in cleaned_sentences]
    tokens_with_trigrams = [trigram_model[bigram_model[sent]] for sent in tokens_with_bigrams]

    #flatten list of sentences for LDA
    all_tokens = [token for sentence in tokens_with_trigrams for token in sentence]
    #corpus for LDA
    dictionary = corpora.Dictionary([all_tokens])
    corpus = [dictionary.doc2bow(text) for text in [all_tokens]]

    #LDA implementation
    num_topics = 3
    lda = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    topics = lda.print_topics(num_words=30)
    for topic in topics:
        print(f"Month-Year: {month_year}")
        print(topic)
    #display 60 relevant terms
    lda_display = gensimvis.prepare(lda, corpus, dictionary, sort_topics=False)
    pyLDAvis.display(lda_display)
    filename = f'ldaTweet_{month_year}.html'
    pyLDAvis.save_html(lda_display, filename)




#WEB SCRAPING
#delete tweet website , keep only non tweet and store in new column
#do web scraping and combine all text data in one column

def fetch_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        return ' '.join(paragraphs)
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return ""

# Tokenization
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
expanded_stopwords = set(stopwords.words('english')).union({'stated', 'going', 'null', "said", "would", "also", "one", "education", "school", "children",
                  "ministry", "sector", "teacher", "teachers", "government", "schools", "kids", "home", "students",
                  "classes", "parents", "child", "staff", "families", "person", "percent", "work", "rain",
                  "year", "since", "last", "group", "whether", "asked", "new", "zealand", "say", "search",
                  "people", "way", "time", "point", "thing", "part", "something", "student", "te", "name", "m", "use",
                  "say", "made", "month", "day", "moe"
                                                            })

#convert date column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format="%d-%b-%Y %I:%M%p")
grouped = df.groupby([df['Date'].dt.year, df['Date'].dt.month])
all_documents = []

for(year, month), group in grouped:
    try:
        print(f"Processing articles from {month}-{year}...")
        # URLs
        urls = group['URL'].tolist()
        # Filter out Twitter URLs
        non_twitter_urls = [url for url in urls if "twitter.com" not in url]

        with ThreadPoolExecutor(max_workers=1000) as executor:
            news_sentences = list(executor.map(fetch_content, non_twitter_urls))

        # Filter out non-English content
        english_news = []
        for news in news_sentences:
            try:
                if detect(news) == 'en':
                    english_news.append(news)
            except LangDetectException:
                pass

        documents = []
        for sentence in english_news:
            tokens = [lemmatizer.lemmatize(token) for token in word_tokenize(sentence.lower()) if token not in expanded_stopwords and token.isalpha()]
            # Consider keeping only nouns for better topic clarity (requires POS tagging)
            tokens = [token for token, pos in nltk.pos_tag(tokens) if pos.startswith('NN')]
            documents.append(tokens)

        # Create LDA model for this month
        dictionary = Dictionary(documents)
        corpus = [dictionary.doc2bow(text) for text in documents]
        lda = LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)

        topics = lda.print_topics(num_words=10)
        for topic in topics:
            print(topic)

        # Generate LDA visualization for this month and save to an HTML file
        lda_display = gensimvis.prepare(lda, corpus, dictionary, sort_topics=False)
        html_filename = f'ldaWeb_{year}_{month}.html'
        pyLDAvis.save_html(lda_display, html_filename)

    except Exception as e:
        print(f"Error processing data for {month}-{year}: {e}")





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















