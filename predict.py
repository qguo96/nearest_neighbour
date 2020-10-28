import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import nltk
from nltk.corpus import stopwords
import numpy as np

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

dev_questions = []
dev_question_answers = []
train_questions = []
train_question_answers = []
filep = os.path.dirname(os.path.abspath(__file__))
train_file = os.path.join(filep, "NQ-open.train.jsonl")
dev_file = os.path.join(filep, "NQ-open.efficientqa.dev.1.1.jsonl")


with open(train_file, "r") as f:
    for line in f:
        d = json.loads(line)
        train_questions.append(d["question"])
        if "answer" not in d:
            d["answer"] = "random"
        train_question_answers.append(d["answer"])

len_train = len(train_questions)

with open(dev_file, "r") as f:
    for line in f:
        d = json.loads(line)
        dev_questions.append(d["question"])
        if "answer" not in d:
            d["answer"] = "random"
        dev_question_answers.append(d["answer"])

len_dev = len(dev_questions)

vectorizer = TfidfVectorizer(stop_words=stop_words)
vectors = vectorizer.fit_transform(dev_questions + train_questions)
dev_vectors = vectors[0:len_dev]

train_vectors = vectors[len_dev:]

dev_predict = [None] * len_dev
for query_index in range(len_dev):
    query_vector = vectors[query_index,:]
    cosine_similarities = linear_kernel(query_vector, train_vectors).flatten()
    dev_predict[query_index] = train_question_answers[np.argmax(cosine_similarities)][0]

print(dev_predict)
