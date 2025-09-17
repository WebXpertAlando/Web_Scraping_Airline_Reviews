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
- Perfrom Topic Modeling.
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

# Sentiment Analysis
### Perform sentiment Analysis
```Python
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity
df['Sentiment'] = df['Cleaned_Review'].apply(get_sentiment)
```

### Categorize sentiment into Positive Neutral Negative
```Python
df['Sentiment_Category'] = df['Sentiment'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
```
