# Csc-46000-Project
CSC 46000 Introduction to Computer Science Project

Group: Daniel Hernandez, Justin Park

## Motivation
The modern day is filled with social media. Almost everyone of all ages today has a social media account and makes constant updates of the changes in their life. 
Every second, approximately 6,000 Tweets are tweeted on Twitter, which corresponds to over 350,000 tweets sent per minute, 500 million tweets per day and around 200 billion tweets per year.
Pew Research Center states that currently, 72% of public uses some type of social media.

As a result of the upsurge of social media and availabilty of phones, Twitter has the potential become an important communication channel in times of emergency.
The availability of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).

But, it’s not always clear whether a person’s words are actually announcing a disaster. Take this example:
<img align="left" width="200" alt="tweet_screenshot" src="https://user-images.githubusercontent.com/22521067/144728661-3e781d06-69b0-4f0f-b466-256c6ee8a152.png">
The author explicitly uses the word “ABLAZE” but means it metaphorically. This is clear to a human right away, especially with the visual aid. But it’s less clear to a machine.

So, with that in mind, we state the question our model will be answering; **Can we predict whether or not a Tweet is talking about a real disaster?**

Building an algorithm that can analyze whether a Tweet is talking about a real disaster can make it possible for first response organizations and agencies to send first response units to that area faster than traditional disaster detection methods. This also allows these agencies to collaborate with phone companies to send alerts to users in the area that a disaster was detected. 

## Data
Our dataset consists of approximately 7600 tweets that have already been classified whether or not the tweet is talking about a real disaster. These tweets have been taken the the Kaggle competition: [Natural Language Processing with Disasters Tweets.](https://www.kaggle.com/c/nlp-getting-started)

### Data Dictionary:
  - **id**: A unique identifier for each tweet
  - **text**: The text of the tweet
  - **location**: The location the tweet was sent from (may be blank)
  - **target**: This denotes whether a tweet is about a real disaster (1) or not (0)

There are only 3 columns that will be of use to use, but there are missing values, and the 'text' columns needs some preprocessing (normalization, tokenization, removing stop words, etc.) before we get started with our models.

## Data Preprocessing and Prelimiary Analysis
Here's how our data looked before any preprocessing:
![Screenshot_2](https://user-images.githubusercontent.com/22521067/144729307-65d3670e-ed8b-4f2b-b22f-dd4fbf96417e.png)

In order to fill in missing values from the 'keyword' column, we obtained a set of unique values in that column, then scanned each entry's respective 'text' value to identify any words that appear in the set of keywords.

In order to fill in missing values from the 'location' column, we used the [Geograpy3 library](https://github.com/somnathrakshit/geograpy3) to extract place names from text.

Cleaning the 'text' column involved removing any html or links in the tweet(they do not provide any context to our problem by themselves), removing punctuation, and putting every character in lowercase.

With clean data, let's visualize how Tweets that are about disasters look compared to normal Tweets.

### Word Cloud

#### Normal Tweets
![normalTweets](https://user-images.githubusercontent.com/22521067/144729565-9a4388c0-3871-4096-8dfd-b744ba72723a.png)

#### Disaster Tweets
![disasterTweets](https://user-images.githubusercontent.com/22521067/144729582-e4cf3d44-ec41-49d5-b4a1-61e159e9fd3e.png)

We can see that there are words that appear more in disaster Tweets and not in normal Tweets. Words like 'fire', 'death' and 'flood'. This makes sense; you would expect Tweets about disasters to use names of disasters in their Tweets.

### Keyword Frequency

#### Normal Tweets
![keywordNormal](https://user-images.githubusercontent.com/22521067/144729765-e0f08113-c8ef-48f0-b51b-78577f313cdc.png)


#### Disaster Tweets
![keywordDisaster](https://user-images.githubusercontent.com/22521067/144729768-d3222698-a4f9-43b0-b06c-d16e21b34934.png)

It seems like there are keywords that both categories share, like 'fire' and 'disaster', but there are still keywords that are in disaster Tweets that are not in normal Tweets, showing that keywords may prove useful in identifying disaster Tweets.

## Model Training and Evaluation

In order to parse our cleaned text in a way our model can understand it, we have to run our text through a word vectorizer to translate Tweets into a matrix of numbers. We will be using TF-IDF as our vectorizer, which measures the originality of a word by comparing the number of times a word appears in a document with the number of documents the word appears in.
![tfIdf](https://user-images.githubusercontent.com/22521067/144763443-0a0d5f45-97f1-4d4e-8108-b8351e80f02d.png)


## Future Work

- Trying a different vectorizer
- Trying to add keyword and locaiton features 
