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



## Evaluation

## Future Work
