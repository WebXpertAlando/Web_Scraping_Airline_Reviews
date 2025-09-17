# Web_Scraping_Airline_Reviews
## Introduction
An airline review is a written or recorded evaluation shared by passengers about their experience with an airline. These reviews usually cover different aspects of the journey, such as:

- Booking & Check-in – ease of ticket purchase, online check-in, boarding process.
- Comfort & Seating – legroom, seat design, cleanliness, overall comfort.
- Cabin Crew Service – professionalism, friendliness, and responsiveness of staff.
- Food & Beverages – quality, variety, and availability of meals/snacks.
- In-flight Entertainment & Amenities – movies, Wi-Fi, magazines, or other features.
- Punctuality & Reliability – on-time departures and arrivals, handling of delays.
- Baggage Handling – efficiency, safety, and accuracy of luggage management.
- Value for Money – whether the experience matched the price paid.

Airline reviews can be posted on travel websites, apps, forums, or social media, and they help future travelers make informed decisions. Airlines themselves also use reviews to improve service quality and customer satisfaction.

# Objectives
- Use python libraries to scrape airline reviews from https://www.airlinequality.com/airline-reviews/air-france.
- Perform data cleaning by removing any unwanted texts.
- Extract Reviews and sentiments by customers from airline quality
- Categorize Sentiment Anaysis into positive, negative and neutral.
- Perform Topic Modeling.
- Display them using wordCloud.
- Distribute Sentiments/Ratings as positive/negatives Using Normal Distibution.

# Python Implementation of web scraping airline Reviews
## Libaries Used:
- Pandas
- Seaborn
- Matplotlib
- selenium
- Beautiful Soup
- nltk
- Scikitlean
- Textblob
- tqdm

# Perform Sentiment Analysis
```Python
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity
df['Sentiment'] = df['Cleaned_Review'].apply(get_sentiment)
```

# Categorize sentiment into Positive Neutral Negative
```Python
df['Sentiment_Category'] = df['Sentiment'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
```

# Perform Topic Modeling
Technique used to automatically discover hidden themes or topics within a large collection of text documents.
Instead of reading each document one by one, topic modeling helps identify what the documents are about by grouping frequently co-occurring words into clusters (topics).
``` Python
vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=2)
X = vectorizer.fit_transform(df['Cleaned_Review'])

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

# Display the top words for each topic
for index, topic in enumerate(lda.components_):
    print(f"Top 10 words for topic #{index}:")
    print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])

# Word Cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Cleaned_Review']))
```
#### Output
```
Top 10 words for topic #0:
['connection', 'did', 'flying', 'je', 'staff', 'flight', 'et', 'france', 'air', 'le']
Top 10 words for topic #1:
['food', 'business', 'paris', 'class', 'luggage', 'service', 'seat', 'flight', 'air', 'france']
Top 10 words for topic #2:
['comfortable', 'verified', 'pleasant', 'flight', 'food', 'cabin', 'seat', 'good', 'cdg', 'crew']
Top 10 words for topic #3:
['paris', 'day', 'told', 'airport', 'staff', 'af', 'airline', 'flight', 'france', 'air']
Top 10 words for topic #4:
['hour', 'airport', 'luggage', 'time', 'hours', 'service', 'paris', 'france', 'air', 'flight']
```


