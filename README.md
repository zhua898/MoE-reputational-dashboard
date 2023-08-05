# MoE-reputational-dashboard
Instruction:
Sentiment analysis pipeline (Example from another project that was analysing tweets)
Step 1: Consolidate data
- Create data frames which consolidate data from all sources. 
- Attributes: Text, Date, Source
- Collect stats on data to understand what you have
 
Step 2: Noise Removal
- Remove noise as usual except stopwords (this is for twitter)
- Lowercase all the content (done)
- Remove Twitter mentions (Example: @user_name)
- Remove URLs (Example: http://xxxx)  done
- Remove common words e.g. nz
- Remove common words in Twitter (Example: "rt", "re", "amp" which refers to retweet, reply and "&")
- Remove punctuation
- Remove non-ASCII characters
- Remove any digits 
 
Step 3: Normalization and remove stop words
- Normalization
- Stem words using Porter's stemming algorithm (stemDocument in tm packages) 
- Normalize the words before replacement of phrasal verbs which is to ensure phrasal verbs in any tenses can be replaced.
 
- Phrasal verbs replacement
- Import a phrasal verb list 
- Examples:
- Replace phrasal verbs by one-word synonyms 
- Remove stopwords using tm
 
Step 4: Sentiment Analysis
- Library (SentimentAnalysis) 
- Sentiment score is calculated by library (syuzhet)
- Could use other algorithms (bing, afinn and nrc) to calculate the sentiment scores
