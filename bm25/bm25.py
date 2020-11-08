import scipy.sparse as sp
import numpy as np
import gc
def compute_corpus_term_idfs(corpus_features, norm_corpus):

    dfs = np.diff(sp.csc_matrix(corpus_features, copy=True).indptr)
    dfs = 1 + dfs # to smoothen idf later
    total_docs = 1 + len(norm_corpus)
    idfs = 1.0 + np.log(float(total_docs) / dfs)
    return idfs


def compute_bm25_similarity(doc_features, corpus_features,
                            corpus_doc_lengths, avg_doc_length,
                            term_idfs, k1=1.5, b=0.75):
    # get corpus bag of words features
    corpus_features = corpus_features.toarray()
    # convert query document features to binary features
    # this is to keep a note of which terms exist per document
    doc_features = doc_features.toarray()[0]
    doc_features[doc_features >= 1] = 1

    # compute the document idf scores for present terms
    doc_idfs = doc_features * term_idfs
    # compute numerator expression in BM25 equation
    numerator_coeff = corpus_features * (k1 + 1)
    numerator = np.multiply(doc_idfs, numerator_coeff)
    print(numerator.shape)
    print("start del")
    del doc_idfs
    del numerator_coeff
    # compute denominator expression in BM25 equation
    denominator_coeff =  k1 * (1 - b +
                                (b * (corpus_doc_lengths /
                                        avg_doc_length)))
    denominator_coeff = np.vstack(denominator_coeff)
    #print(corpus_features.shape)
    #print(denominator_coeff.shape)
    denominator = corpus_features + denominator_coeff
    del denominator_coeff
    #print(denominator.shape)
    # compute the BM25 score combining the above equations
    #gc.collect()
    np.divide(numerator, denominator, out = denominator)
    #del numerator
    bm25_scores = np.sum(denominator,axis=1)
    #del denominator
    print(bm25_scores)
    print("reach end")
    return bm25_scores
