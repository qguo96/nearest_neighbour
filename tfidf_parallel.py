from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words
import json
import nltk
import os
import string
import numpy as np
import copy
import pandas as pd
import pickle
import re
import math
from tqdm import tqdm

def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def convert_lower_case(data):
    return np.char.lower(data)

def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

def remove_apostrophe(data):
    return np.char.replace(data, "'", "")

def stemming(data):
    stemmer= PorterStemmer()

    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text


def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data) #remove comma seperately
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = convert_numbers(data)
    data = stemming(data)
    data = remove_punctuation(data)
    data = convert_numbers(data)
    data = stemming(data) #needed again as we need to stem the words
    data = remove_punctuation(data) #needed again as num2word is giving few hypens and commas fourty-one
    data = remove_stop_words(data) #needed again as num2word is giving stop words 101 - one hundred and one
    return data


filep = os.path.dirname(os.path.abspath(__file__))
#train_file = os.path.join(filep, "NQ-open.train.jsonl")
#dev_file = os.path.join(filep, "NQ-open.efficientqa.dev.1.1.jsonl")
train_file = os.path.join(filep, "test_train.jsonl")
dev_file = os.path.join(filep, "test_dev.jsonl")

dev_questions = []
dev_question_answers = []
train_questions = []
train_question_answers = []
with open(train_file, "r") as f:
    for line in f:
        d = json.loads(line)
        train_questions.append(_normalize_answer(d["question"]))
        if "answer" not in d:
            d["answer"] = "random"
        train_question_answers.append(d["answer"])

len_train = len(train_questions)
#print(len_train)


with open(dev_file, "r") as f:
    for line in f:
        d = json.loads(line)
        dev_questions.append(_normalize_answer(d["question"]))
        if "answer" not in d:
            d["answer"] = "random"
        dev_question_answers.append(d["answer"])

len_dev = len(dev_questions)

#print(train_questions)
#print(train_questions[:28])

processed_text = []
#processed_title = []

#for question in tqdm(train_questions):
#    print(question)
#exit()

#print(type(train_questions))
#index = 0
for question in train_questions:
    #print(index)
    #index += 1
    processed_text.append(word_tokenize(str(preprocess(question))))
#    processed_title.append(word_tokenize(str(preprocess(i[1]))))
#print(processed_text)#len is 28
#print(len(processed_text))
DF = {}
for i in range(len_train):
    tokens = processed_text[i]
    for w in tokens:
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}

for i in DF:
    DF[i] = len(DF[i])

#print(DF)
#exit()
total_vocab_size = len(DF)
total_vocab = [x for x in DF]
#print(total_vocab_size)


def doc_freq(word):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c


doc = 0
tf_idf = {}
N = len_train
for i in range(N):
    tokens = processed_text[i]
    counter = Counter(tokens)
    words_count = len(tokens)
    for token in np.unique(tokens):
        tf = counter[token]/words_count
        df = doc_freq(token)
        idf = np.log((N+1)/(df+1)) #numerator is added 1 to avoid negative values
        tf_idf[doc, token] = tf*idf
    doc += 1

#print(tf_idf)
#exit()

def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    cos_sim = np.nan_to_num(cos_sim)
    return cos_sim

D = np.zeros((N, total_vocab_size))
for i in tf_idf:
    ind = total_vocab.index(i[1])
    D[i[0]][ind] = tf_idf[i]

#print(D[0])
def gen_vector(tokens):
    Q = np.zeros((len(total_vocab)))
    counter = Counter(tokens)
    words_count = len(tokens)
    for token in np.unique(tokens):
        tf = counter[token]/words_count
        df = doc_freq(token)
        #print(df)
        idf = math.log((N+1)/(df+1))
        try:
            ind = total_vocab.index(token)
            Q[ind] = tf*idf
        except:
            pass
    return Q

"""
def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    cos_sim = np.nan_to_num(cos_sim)
    return cos_sim
"""
def cosine_similarity(query):
    preprocessed_query = preprocess(query)
    tokens = word_tokenize(str(preprocessed_query))
    d_cosines = []
    query_vector = gen_vector(tokens)
    #print(query_vector)
    #print(query_vector.shape)
    #print(D.shape) # 28,130
    for d in D:
        #print(d.shape) 130
        d_cosines.append(cosine_sim(query_vector, d))
    #print(d_cosines)
    print(query_vector.shape[0])
    #exit()
    print(D.shape)
    num_words, set_num = D.shape
    print(num_words)
    exit()
    mproduct = np.linalg.norm(D, axis=1) * (np.linalg.norm(query_vector)) #
    mdivide = np.nan_to_num(np.divide((D.dot(query_vector.reshape(130, 1))).reshape(28,), mproduct))
    #out = np.array(d_cosines).argsort()[-1:][::-1]
    out = np.array(mdivide).argsort()[-1:][::-1]
    return out[0]

#     for i in out:
#         print(i, dataset[i][0])

vectors = [None] * len_dev
for query_index in range(len_dev):
    #if query_index % 20 == 0:
    #    print(query_index)
    query = dev_questions[query_index]
    preprocessed_query = preprocess(query)
    tokens = word_tokenize(str(preprocessed_query))
    #d_cosines = []
    query_vector = gen_vector(tokens)
    #if query_index == 0:
        #print(query_vector)
    vectors[query_index] = query_vector
    #nearest_index = cosine_similarity(dev_questions[query_index])
    #dev_predict[query_index] = train_question_answers[nearest_index][0]
vectors = np.array(vectors) # 19, 130
num_dev, _ = vectors.shape
num_train, set_num = D.shape

#print(num_words)
vector_norms = np.linalg.norm(vectors, axis=1)
#print(vectors.shape)
#print(vector_norms.reshape(19,1))
mproduct_rev = (vector_norms.reshape(num_dev,1)).dot((np.linalg.norm(D, axis=1)).reshape(1,num_train)) #(19, 28)
mdivide_rev = np.nan_to_num(np.divide(vectors.dot(D.T), mproduct_rev))

dev_predict = [None] * len_dev
for query_index in range(len_dev):
    #print(mdivide_rev[i].shape)
    #exit()
    nearest_index = int(mdivide_rev[query_index].argsort()[-1:][::-1])
    dev_predict[query_index] = train_question_answers[nearest_index][0]

predict_output = [None] * len_dev
for i in range(len_dev):
    output_dict = {'question': dev_questions[i],
            'prediction': dev_predict[i]
            }
    predict_output[i] = output_dict


pred_file = os.path.join(filep, 'tfidf_test_predict.json')
with open(pred_file, 'w') as output:
    output.write(json.dumps(predict_output, indent=4) + '\n')
