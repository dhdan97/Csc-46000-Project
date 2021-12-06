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

Before working with our models; let us take note of the balance of data:
![Screenshot_3](https://user-images.githubusercontent.com/22521067/144773442-06c09a57-f4d6-4054-97b3-2acbb2d08c16.png)
It seems like we have a pretty balanced set of data between postive and negative cases. This is good, since imbalance data can lead to our models doing a worse job at predicting the cateogory with less data for it.

Now we are ready to feed our data into our models. We first start with Logistic Regression:
![Screenshot_3](https://user-images.githubusercontent.com/22521067/144773996-42cfcdd6-f147-4845-a4e2-31074d8fa29d.png)

We can visualize our models performance with a ROC curve:

![Screenshot_5](https://user-images.githubusercontent.com/22521067/144786410-122c99be-71d5-425a-857a-8b3199e7abcf.png)

74% Accuracy and a ROC AUC score of 0.8; Pretty decent! Let's see if we can improve our scores with some hyperparameter tuning using GridSearchCV.
We supply GridSearchCV with a set of parameters and their values to iterate through, and GridSearchCV will exhuastively search through each combination of parameters and find the set of parameters that returns the best score:
![Screenshot_6](https://user-images.githubusercontent.com/22521067/144786483-23d92a31-b6f5-4b61-9fbd-c490d75643b7.png)

It seems like hyperparameter tuning has not made a significant improvement to our model; at least with the set of parameters we gave.

Let's try using a different model; Naive Bayes:

![Screenshot_7](https://user-images.githubusercontent.com/22521067/144786505-bf9133ee-2fb0-4fed-ab28-acb6e129638d.png)

We can check which value of alpha(a hyperparameter) gives us a better score:

![Screenshot_8](https://user-images.githubusercontent.com/22521067/144786542-2726b0df-a915-4b81-a066-e4fa80389298.png)

It seems like Naive Bayes has a similar level of accuracy with our logisitic regression model. But with an accuracy of 74%, we believe this model has promise and should prove effective for further development and use.

## Future Work

- **Trying a different vectorizer:** Different vectorizers that provide different interpretations of words from numbers might change our model's performance. Gensim library is a vectorizer that using topic model rather than word frequency.

- **Trying to add keyword and locaiton features into our model:** While these columns were useful in our preliminary analysis, we we're not able to able to successful add these features into our models due to the interest of time. A probable plan involves one-hot encoding these columns into binary features, and appending these features into the matrix that is produced from the word vectorization of our Tweet.

- **Adjust our true positive rate/false positive rate based on our model's goal:** Based on the specific use case for our model, we would want to prioritize either our true positive rate or our true negative rate. Suppose we would send out a full first response team to the area based on whether if a Tweet was talking about a disaster. In this case we would want to focus on true positive rates; and we would make the necessary adjustments to ensure we correctly detect disaster Tweets, as opposed to ensuring we detect normal Tweets. 
