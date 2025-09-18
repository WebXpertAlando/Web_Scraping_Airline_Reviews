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
- Display Topics Using WordCloud.
- Visualize Sentiments Using Normal Distibution.
- Visualize Ratings Using Boxplots.

# Python Implementation of Web Scraping Airline Reviews
## Libaries Used:
- Pandas.
- Seaborn.
- Matplotlib.
- selenium.
- Beautiful Soup.
- nltk.
- Scikitlean.
- Textblob.
- tqdm.

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

# Display Topics Using WordCloud
A wordcloud is a visual representation of text data where:
- Words appear larger or bolder based on how frequently they occur in the text.
- It’s often used to quickly identify key themes or topics in reviews, surveys, or any large text dataset.
- Colors and orientations are sometimes varied to make it visually engaging.
So in this project, we have generated a word cloud from airline reviews, words like “service”, “delay”, “comfortable”, and “crew” might appear bigger depending on how often they’re mentioned.

```Python
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Cleaned_Review']))
```
### Plot the WordCloud
```
#plot the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```
#### Output

<img width="793" height="397" alt="Screenshot at 2025-09-18 11-17-20" src="https://github.com/user-attachments/assets/3ab4da60-f562-4707-b6a0-7da4a55cbedf" />

# Visualize Sentiments as Positive Negative and Neutral
```Python
# Sentiment Distribution Plot
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Sentiment_Category', palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
```

#### Output
<img width="846" height="475" alt="Screenshot at 2025-09-18 11-53-48" src="https://github.com/user-attachments/assets/e5362759-8ac3-41ae-830c-43aa1a64f271" />

As we can see we have visualized the sentiments as positive, negative and neutral. It can be seen that the reviews that have been categorized as postive are high as compared to those of negative reviews. This shows that the airline can still get more customers in the future and get positive sentiments. 

# Visualize Ratings
```Python
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Sentiment_Category', y='Rating', palette='viridis')
plt.title('Rating Distribution by Sentiment')
plt.xlabel('Sentiment Category')
plt.ylabel('Rating')
plt.show()
```
#### Output
<img width="862" height="487" alt="Screenshot at 2025-09-18 12-13-20" src="https://github.com/user-attachments/assets/a9d4569b-a135-42cb-ab9e-1ca7ff865928" />
The boxplots shows how positive ratings are high as compared to negative and neutral. 
