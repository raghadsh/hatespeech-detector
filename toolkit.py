# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:53:25 2019

@author: raghe
"""
import demoji
import re
import pandas
import emoji
import pickle

"""" GENERAL HELPING METHODS"""
def open_excel(fileFullPath, index = None):
    
    if index is not None: 
        df = pandas.read_excel(fileFullPath)
    else:
        df = pandas.read_excel(fileFullPath)
    
    return df

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def translate(text, conversion_dict, before=None):
    """
    
    Translate words from a text using a conversion dictionary - https://stackoverflow.com/a/48953324/950520

    Arguments:
        text: the text to be translated
        conversion_dict: the conversion dictionary
        before: a function to transform the input
        (by default it will to a lowercase)
    """
    # if empty:
    if not text: return text
    # preliminary transformation:
    before = before or str.lower
    t = before(text)
    for key, value in conversion_dict.items():
        t = t.replace(key, " " + value)
    return t

"""" END GENERAL HELPING METHODS"""

""" PREPROCESS METHODS AND PATTERNS """

""" regular expression patterns"""
email = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\.?[a-zA-Z0-9-.]*"
linksPttren = r"http[s]?:?//(?:[a-zA-Z]|[0-9]|[$-_@.#&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+" 
mentions = r"@[a-zA-Z0-9_.-]+"
atSign = r"@"
hashSign = r"#"
UnderscoreSign = r"_"
ENGLISH_CHARS =r'[a-zA-Z]+\b(?<!urllink|mention)'
p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')

def download_emoji():
    demoji.download_codes()

def remove_cosuctive_letters(text):
    text = re.sub(r'([^a-zA-Z0-9])\1+\1+', r'\1', text)  
    return text

def replace_emoji_with_description(text):  
    dic = demoji.findall(text)
    text = translate(text, dic) 
    return text

def replace_emoji_with_unifiedword(text):  
    text = ''.join(['EM' if c in emoji.UNICODE_EMOJI else c for c in text]) # replace the emoji with EM
    return text

def normlization(text):
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    return text
    
def deNoise(text):
    noise = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
    text = re.sub(noise, '', text)
    return text

def normlize_twitter_spesfice_tokens(text):
    text =  re.sub(mentions, 'mention', text) # replace  mentions with mention 
    text =  re.sub(atSign, ' ', text) # remove @
    text =  re.sub(hashSign, ' ', text) #remove #
    text = re.sub(UnderscoreSign,' ',text) #remove _
    text = re.sub(linksPttren, 'urllink', text) # remove all links and replace them with "link"
    return text

def remove_non_arabic_letteres(text):
    text = re.sub(ENGLISH_CHARS, "",  text)
    
    return text
	
def cleaning(text):
    text = re.sub(r'[.,\"،\/#!$%\^&\*;:{}=\-_`~()?؟﴾﴿]',' ',text) #remove punctoation
    text = re.sub(r'[0-9]+',' ',text) #remove numbers
    text = re.sub(r'[\u0660-\u0669]+',' ',text) #remove ar numbers
    text = re.sub('\s+', ' ', text).strip() #remove whitspaces
    #text = " ".join(re.split("[^\w]*",text)).strip()
    return text

def remove_hashtags(text,list_of_hashtags):
    """
    Remove hashtags from the text if they exist in hashtags
    """
    text_hashtags = re.findall(r"(#\w+)", text)
    for hashtag in text_hashtags:
    
        if hashtag in list_of_hashtags:
            text = re.sub(hashtag,'',text)
    
    return text 
    
def remove_stopwords(text, stopwords):
    
    word_list = text.split(' ')
    filtered_words = [word for word in word_list if word not in stopwords]
    text = ' '.join(filtered_words)
    
    return text 
    
""" END PREPROCESS METHODS AND PATTERNS """

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    plt.savefig(title +'.png')
    

def save_report(report,name):
    df = pandas.DataFrame(report).transpose()
    df.to_excel(r''+name+'.xlsx')
    