
from normalization import normalize_corpus
from utils import build_feature_matrix
from bm25 import compute_corpus_term_idfs
from bm25 import compute_bm25_similarity
from semantic_similarity import sentence_similarity
import numpy as np
import os
import json
import sys

def run():
    """
    answers=['Functions are used as one-time processing snippet for inling and jumbling the code.',
    'Functions are used for reusing, inlining and jumbling the code.',
    'Functions are used as one-time processing snippet for inlining and organizing the code.',
    'Functions are used as one-time processing snippet for modularizing and jumbling the code.',
    'Functions are used for reusing, inling and organizing the code.',
    'Functions are used as one-time processing snippet for modularizing and organizing the code.',
    'Functions are used for reusing, modularizing and jumbling the code.',
    'Functions are used for reusing, modularizing and organizing the code.']

    model_answer = ["Functions are used for reusing, modularizing and organizing the code."]
    """
    dev_questions = []
    dev_question_answers = []
    train_questions = []
    train_question_answers = []
    filep = os.path.dirname(os.path.abspath(__file__))
    #train_file = os.path.join(filep, "NQ-open.train.jsonl")
    #dev_file = os.path.join(filep, "NQ-open.efficientqa.dev.1.1.jsonl")
    train_file = os.path.join(filep, "test_train.jsonl")
    dev_file = os.path.join(filep, "test_dev.jsonl")

    with open(train_file, "r") as f:
        for line in f:
            d = json.loads(line)
            train_questions.append((d["question"]))
            if "answer" not in d:
                d["answer"] = "random"
            train_question_answers.append(d["answer"])

    len_train = len(train_questions)

    with open(dev_file, "r") as f:
        for line in f:
            d = json.loads(line)
            dev_questions.append((d["question"]))
            if "answer" not in d:
                d["answer"] = "random"
            dev_question_answers.append(d["answer"])

    len_dev = len(dev_questions)

    answers = train_questions
    model_answer = dev_questions

    # normalize answers
    norm_corpus = normalize_corpus(answers, lemmatize=True)
    print(sys.getsizeof(norm_corpus))
    print(len(norm_corpus))
    # normalize model_answer
    norm_model_answer =  normalize_corpus(model_answer, lemmatize=True)

    vectorizer, corpus_features = build_feature_matrix(norm_corpus,feature_type='frequency')

    # extract features from model_answer
    model_answer_features = vectorizer.transform(norm_model_answer)

    doc_lengths = [len(doc.split()) for doc in norm_corpus]
    avg_dl = np.average(doc_lengths)
    corpus_term_idfs = compute_corpus_term_idfs(corpus_features, norm_corpus)

    train_predict = [None] * len_dev
    dev_predict = [None] * len_dev
    for index, doc in enumerate(model_answer):
        print(index)
        doc_features = model_answer_features[index]
        #bm25_scores = compute_bm25_similarity(model_answer_features,corpus_features,doc_lengths,avg_dl,corpus_term_idfs,k1=0.82, b=0.68)
        bm25_scores = compute_bm25_similarity(doc_features,corpus_features,doc_lengths,avg_dl,corpus_term_idfs,k1=0.82, b=0.68)
        exit()
        semantic_similarity_scores=[]
        for sentence in answers:
            score=(sentence_similarity(sentence,model_answer[0])+sentence_similarity(model_answer[0],sentence))/2
            semantic_similarity_scores.append(score)
        doc_index=0
        max_index = 0
        max_score = 0
        for score_tuple in zip(semantic_similarity_scores,bm25_scores):
            sim_score=((score_tuple[0]*10)+score_tuple[1])/2
            if sim_score > max_score:
                max_score = sim_score
                max_index = doc_index
            doc_index=doc_index+1
        dev_predict[index] = train_question_answers[max_index][0]
    predict_output = [None] * len_dev
    for i in range(len_dev):
        output_dict = {'question': dev_questions[i],
                'prediction': dev_predict[i]
                }
        predict_output[i] = output_dict

    pred_file = os.path.join(filep, 'ef_dev_predict.json')
    with open(pred_file, 'w') as output:
        output.write(json.dumps(predict_output, indent=4) + '\n')

run()
