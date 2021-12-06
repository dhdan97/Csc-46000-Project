import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk import word_tokenize
import spacy
import locationtagger
import re

import nltk
# essential entity models downloads
nltk.downloader.download('maxent_ne_chunker')
nltk.downloader.download('words')
nltk.downloader.download('treebank')
nltk.downloader.download('maxent_treebank_pos_tagger')
nltk.downloader.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.downloader.download('maxent_ne_chunker')
nlp = spacy.load("en_core_web_sm")

# Create DataFrame
train_df = pd.read_csv(
    'E:\\E_drive\\E_drive_desktop\\school\\2021_Fall\\CSC 460\\project\\github\\Csc-46000-Project\\Project\\train.csv')

# create word vector which contains the dissaster keywords
DISSASTER_VECTOR = set(train_df['keyword'].unique())


def fillKey(text, str='str'):
    """function which will fill in missing values for keyword column dependent on text column

    Args:
        text ([string]): [tweets to be processed]
        str ([string]): [indicator for whether to return string of keywords delimited by '%20' or to return list]

    Returns:
        [String]: [returns string delimited by '%20' containing keywords of tweet]
    """
    # tokenize the text and store into text_vector
    text_vector = set(word_tokenize(text.lower()))
    # find the dissaster keywords in text_vector
    filler = list(set.intersection(DISSASTER_VECTOR, text_vector))
    if str == 'str':
        # combine the filler array with a delimiter matching that of original format from keyword column
        filler = '%20'.join(filler)
    else:
        filler = np.array(filler)
    return filler


def fillTagLoc(txt):
    """function which will fill in missing values for location column dependent on text column

    Args:
        text ([string]): [tweets to be processed]

    Returns:
        [String]: [returns string delimited by '%20' containing locations of tweet]
    """
    # using regex to handle # and @ characters
    txt = re.sub(r'[#|@]', r'', txt)
    # find_locations uses the text to find the locations
    place_entity = locationtagger.find_locations(text=txt)
    # combine the filler array with a delimiter matching that of original format from location column
    filler = '%20'.join(set(place_entity.countries +
                            place_entity.regions + place_entity.cities))
    return filler


def transformDF(df, func):
    """function transforms a copy of df's column, string using a function, func

    Args:
        df ([DataFrame]): [DataFrame to use to transform]
        func ([function]): [function or method used to transform data of df]

    Returns:
        [DataFrame]: [returns the transformed DataFrame]
    """
    # auxilery DataFrame to handle DataFrame.apply()
    aux_df = pd.DataFrame()
    aux_df = df.apply(func)
    return aux_df


def cleankeyword(data, column):
    """function to convert keyword column into np.array and create another column with np.array format

    Args:
        data (DataFrame): DataFrame to be used in cleaning column
        column (string): column name
    """
    str1 = ' '
    final_string = []
    for sent in data[column]:
        str1 = sent.split("%20")
        final_string.append(str1[0])
    data['clean_keyword'] = np.array(final_string)


def cleanhtml(sentence):
    """function to remove http

    Args:
        sentence (string): removes http from sentence

    Returns:
        string: returns the modified sentence
    """
    # using regex to remove sub string
    cleantext = re.sub(r'http\S+', r'', sentence)
    return cleantext


def cleanpunc(sentence):
    """function to remove punctuations

    Args:
        sentence (string): removes punctuations from sentence

    Returns:
        string: returns the modified sentence
    """
    # using regex to remove sub string
    cleaned = re.sub(r'[?|!|\'|"|#@_%$\n:"]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|)|\|/]', r' ', cleaned)
    cleaned = re.sub(r'[0-9]', r'', cleaned)
    cleaned = re.sub(r'[ ]{2,}', r' ', cleaned)
    cleaned = re.sub(r'amp', r'', cleaned)
    return cleaned

# function for cleaning 'column' of dataframe 'data' and saving cleaned text in column 'clean_txt'


def cleantxt(data, column):
    """function to clean column, 'column' from DataFrame, 'data'

    Args:
        data (DataFrame): DataFrame to be used in cleaning column
        column (string): column name
    """
    str1 = ' '
    final_string = []
    for sent in data[column]:
        rem_html = cleanhtml(sent)
        rem_punc = cleanpunc(rem_html)
        str1 = rem_punc.lower()
        final_string.append(str1)
    data['clean_txt'] = np.array(final_string)
