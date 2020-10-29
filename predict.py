import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import nltk
from nltk.corpus import stopwords
import numpy as np
import string
import re

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

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

dev_questions = []
dev_question_answers = []
train_questions = []
train_question_answers = []
filep = os.path.dirname(os.path.abspath(__file__))
#train_file = os.path.join(filep, "NQ-open.train.jsonl")
train_file = os.path.join(filep, "cat_train.jsonl")
dev_file = os.path.join(filep, "NQ-open.efficientqa.dev.1.1.jsonl")


with open(train_file, "r") as f:
    for line in f:
        d = json.loads(line)
        train_questions.append(_normalize_answer(d["question"]))
        if "answer" not in d:
            d["answer"] = "random"
        train_question_answers.append(d["answer"])

len_train = len(train_questions)

with open(dev_file, "r") as f:
    for line in f:
        d = json.loads(line)
        dev_questions.append(_normalize_answer(d["question"]))
        if "answer" not in d:
            d["answer"] = "random"
        dev_question_answers.append(d["answer"])

len_dev = len(dev_questions)

"""
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
"""
vectorizer = TfidfVectorizer(stop_words=stop_words)
vectors = vectorizer.fit_transform(dev_questions + train_questions)
dev_vectors = vectors[0:len_dev]

train_vectors = vectors[len_dev:]

dev_predict = [None] * len_dev
for query_index in range(len_dev):
    query_vector = vectors[query_index,:]
    cosine_similarities = linear_kernel(query_vector, train_vectors).flatten()
    dev_predict[query_index] = train_question_answers[np.argmax(cosine_similarities)][0]


predict_output = [None] * len_dev
for i in range(len_dev):
    output_dict = {'question': dev_questions[i],
            'prediction': dev_predict[i]
            }
    predict_output[i] = output_dict


pred_file = os.path.join(filep, 'ef_dev_predict.json')
with open(pred_file, 'w') as output:
    output.write(json.dumps(predict_output, indent=4) + '\n')
