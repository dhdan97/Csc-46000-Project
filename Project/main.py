from sys import displayhook
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import data_method


# Create DataFrame
df = pd.read_csv(
    'E:\\E_drive\\E_drive_desktop\\school\\2021_Fall\\CSC 460\\project\\github\\Csc-46000-Project\\Project\\train.csv')


def location_transform():
    """[Method separated because it takes too long to render when running with main.]
    """
    # Filling in missing values for location column. Applying fillna() since some locations are user's locations.
    df['location'].fillna(data_method.transformDF(
        df['text'], data_method.fillTagLoc), inplace=True)


def dissaster_freq_vis():
    """function to show model which shows the frequency of dissaster keywords in tweets when actual dissaster and not dissaster
    """
    # prepping 'clean_keyword' column for model
    data_method.cleankeyword(df, 'keyword')
    # creating plots
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))
    fig.tight_layout(pad=10.0)
    # bool_bin: 0 for not dissaster, 1 for dissaster
    bool_bin = 0
    # populating plt with data
    while bool_bin < 2:

        if bool_bin == 0:
            title_str = 'Normal Tweets'
        else:
            title_str = 'Disaster Tweets'

        temp = pd.DataFrame(df[df['target'] == bool_bin].groupby(
            'clean_keyword')['id'].count())
        temp.sort_values('id', ascending=False, inplace=True)
        ax = sns.barplot(temp.index[1:10], temp['id']
                         [1:10], ax=axes[bool_bin])
        axes[bool_bin].set_title(title_str, fontsize=16, fontweight='bold')
        axes[bool_bin].set_ylabel('Frequency', fontweight='bold', fontsize=16)
        axes[bool_bin].set_xlabel('Keyword', fontweight='bold', fontsize=16)
        bool_bin += 1

    plt.show()


def word_cloud_vis(bool_bin):
    """This function shows data visualization for word cloud from tweets

    Args:
        bool_bin (integer): int of 0 or 1, 0 for not dissaster, 1 for dissaster
    """
    # prepping 'clean_txt' column by cleaning the tweets in 'text' column
    data_method.cleantxt(df, 'text')
    # populating all_words for word cloud
    all_words = ' '.join(
        [text for text in df['clean_txt'][df['target'] == bool_bin]])

    wordcloud = WordCloud(width=800, height=500, random_state=21,
                          max_font_size=110).generate(all_words)

    plt.figure(figsize=(16, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()


def txt_model():
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.90, min_df=2, max_features=300, stop_words='english')
    # max_df defines the frequency cutoff for words in the document;
    # so we will remove words that appear in more that 90% of the document. Why? Because words that appear close to 100% of all Tweets
    # will not help in deciding whether a Tweet is about a disaster or not
    # Same idea with min_df, but for words that appear too infrequently
    # max_features is self explaining, but why? To prevent overfitting by reducing the complexity of the model
    # TF-IDF feature matrix
    tfidf = tfidf_vectorizer.fit_transform(df['clean_txt'])
    print('tfidf.shape: ', tfidf.shape)
    print('tfidf[10]: ', tfidf[10])
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf, df['target'], test_size=0.30, random_state=42)
    # Logistic Regression
    # creating Logreg object and fitting our data to it
    logreg = LogisticRegression()
    logreg = logreg.fit(X_train, y_train)
    # evaluating our model performance
    y_pred = logreg.predict(X_test)
    print("Accuracy of logistic regression classifier: ",
          logreg.score(X_test, y_test))
    print("confusion matrix: ", confusion_matrix(y_test, y_pred))
    print('recall score: ', recall_score(y_test, y_pred))
    print('precision score: ', precision_score(y_test, y_pred))

    # Now to create ROC curve
    # generate the probabilities
    y_pred_prob = logreg.predict_proba(X_test)[:, 1]
    # calculate the roc metrics
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    # Plot the ROC curve
    plt.plot(fpr, tpr)
    # Add labels and diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve [Logistic Regression]")
    plt.plot([0, 1], [0, 1], "k--")
    plt.show()
    print('ROC AUC score: ', roc_auc_score(y_test, y_pred_prob))
    # 74% Accuracy and a 0.80 ROC area under curve score; not bad. Let's try some hyperparameter tuning
    # dictionary of parameters and their values to iterate through to find the best combination
    params = {'solver': ['liblinear', 'lbfgs'], 'max_iter': [
        100, 200, 300, 400], 'tol': [0.01, 0.001, 0.0001, 0.00001]}
    # Instantiate GridSearchCV with the required parameters
    grid_model = GridSearchCV(estimator=logreg, param_grid=params, cv=5)
    # Fit data to grid_model
    grid_model_result = grid_model.fit(tfidf, df['target'])
    # Summarize results
    best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
    print("Best: %f using %s" % (best_score, best_params))
    # creating Logreg object and fitting our data to it
    logreg = LogisticRegression(max_iter=200, solver='liblinear', tol=0.001)
    logreg = logreg.fit(X_train, y_train)
    # evaluating our model performance
    y_pred = logreg.predict(X_test)
    y_pred_new_threshold = (logreg.predict_proba(X_test)[
                            :, 1] >= 0.85).astype(int)
    print("Accuracy of logistic regression classifier: ",
          logreg.score(X_test, y_test))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("confusion matrix with 0.85 threshold:\n",
          confusion_matrix(y_test, y_pred_new_threshold))


def train_and_predict(alpha, X_train, X_test, y_train, y_test):
    # Instantiate the classifier: nb_classifier
    nb_classifier = MultinomialNB(alpha=alpha)
    # Fit to the training data
    nb_classifier.fit(X_train, y_train)
    # Predict the labels: pred
    pred = nb_classifier.predict(X_test)
    # Compute accuracy: score
    score = accuracy_score(y_test, pred)
    return score


def naive_bayes_model():
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.90, min_df=2, max_features=300, stop_words='english')
    # max_df defines the frequency cutoff for words in the document;
    # so we will remove words that appear in more that 90% of the document. Why? Because words that appear close to 100% of all Tweets
    # will not help in deciding whether a Tweet is about a disaster or not
    # Same idea with min_df, but for words that appear too infrequently
    # max_features is self explaining, but why? To prevent overfitting by reducing the complexity of the model
    # TF-IDF feature matrix
    tfidf = tfidf_vectorizer.fit_transform(df['clean_txt'])
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf, df['target'], test_size=0.30, random_state=42)
    # Instantiate a Multinomial Naive Bayes classifier: nb_classifier
    nb_classifier = MultinomialNB()
    # Fit the classifier to the training data
    nb_classifier.fit(X_train, y_train)
    # Create the predicted tags: pred
    pred = nb_classifier.predict(X_test)
    # Calculate the accuracy score: score
    score = accuracy_score(y_test, pred)
    print('accuracy score', score)
    # Calculate the confusion matrix: cm
    cm = confusion_matrix(y_test, pred)
    print('confusion matrix: ', cm)
    # improving Naive Bayes model
    # Create the list of alphas: alphas
    alphas = np.arange(0, 1, 0.1)

    # Iterate over the alphas and print the corresponding score
    for alpha in alphas:
        print('Alpha: ', alpha)
        print('Score: ', train_and_predict(
            alpha, X_train, X_test, y_train, y_test), '\n')


def main():
    # Count the amount of rows and columns
    print("Shape of train_df:", df.shape)
    print('\n\n')
    # Counting the amount of missing values
    print("null values:\n", df.isnull().sum())
    print('\n\n')
    # Observing the DataFrame's summary
    df.info()

    # Filling in missing values for keyword column
    df['keyword'] = data_method.transformDF(df['text'], data_method.fillKey)

    # Filling in missing values for location column
    # this method will be commented out for now as it takes 51 minutes on Visual Studio Code to run or 18 minutes on Jupyter Notebook to run
    # t0 = time.time()
    # location_transform()
    # t1 = time.time()
    # print("Time taken to clean location column: ", t1-t0)

    # showing the frequency visualizations
    dissaster_freq_vis()

    # showing the word cloud visualizations of tweets
    # not dissaster tweets
    word_cloud_vis(0)
    # dissaster tweets
    word_cloud_vis(1)

    # Models time
    print('Model created from tweets: \n')
    txt_model()

    print('Naive Bayes model created from tweets: \n')
    naive_bayes_model()


main()
