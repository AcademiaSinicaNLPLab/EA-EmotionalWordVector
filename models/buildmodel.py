# -*- coding: utf8 -*-
import sys, os
sys.path.append("../")
import cPickle as pickle
import numpy as np
from scipy import spatial
from gensim.models import word2vec
from gensim.models import Word2Vec
import pymongo
from pymongo import MongoClient
import logging
from graph.buildgraph import shortestPath



def build_word_vector(w2v_sentences,model_save_path,num_features,min_word_count,context):

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # Set values for various parameters
    # num_features = 300    # Word vector dimensionality                      
    # min_word_count = 50   # Minimum word count                        
    num_workers = 6       # Number of threads to run in parallel
    # context = 10          # Context window size                                                                                    
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print "Training model..."
    model = word2vec.Word2Vec(w2v_sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling)

    # If you don't plan to train the model any further, calling 
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and 
    # save the model for later use. You can load it later using Word2Vec.load()
    model.save(model_save_path)


def calculate_similarity_score_using_path(path, edge_score, path_graph=None, threshold=None):
    if threshold: assert path_graph != None
    if path:
        # print path
        if len(path) == 1:
            return 1.
        else:
            score = 1.
            for w1, w2 in zip(path,path[1:]):
                if threshold:
                    # print w1,w2,path_graph[w1][w2]
                    if path_graph[w1][w2] > threshold:
                        return 0.
                score *= edge_score[w1][w2]
            return score
    else:
        return 0.

def similarity_between_words(word1, word2, model):
    return (2.0 - spatial.distance.cosine(model[word1], model[word2]))/2.0

def calculate_score_on_two_pole_axis(score_p, score_n):
    # print score_p, score_n
    if score_p == 0 and score_n == 0:
        return 0.
    else:
        score_on_axis = 1 - 2*score_n/(score_n + score_p)
        return score_on_axis

def build_axis(poles):
    import itertools
    axis = list(itertools.combinations(poles,2))
    print len(axis)
    return axis

def map_word_on_two_pole_axis_for_model(word, axis, model):
    result = np.zeros(len(axis))
    for i,(pole_p, pole_n) in enumerate(axis):
        score_p = similarity_between_words(word, pole_p, model)
        score_n = similarity_between_words(word, pole_n, model)
        result[i] = calculate_score_on_two_pole_axis(score_p, score_n)
        # print word,pole_p,pole_n,result[i]
    return result

def map_word_on_two_pole_axis_for_graph(word, axis, score_graph, path_graph):
    result = np.zeros(len(axis))
    for i,(pole_p, pole_n) in enumerate(axis):
        if word in path_graph:
            path_p = shortestPath(path_graph,word,pole_p)
            path_n = shortestPath(path_graph,word,pole_n)
            score_p = calculate_similarity_score_using_path(path_p, score_graph)
            score_n = calculate_similarity_score_using_path(path_n, score_graph)
            result[i] = calculate_score_on_two_pole_axis(score_p, score_n)
            print word,pole_p,pole_n,result[i]
    return result    

def map_word_on_one_pole_axis(word, axis, score_graph, path_graph, threshold=None):
    result = np.zeros(len(axis))
    for i,pole in enumerate(axis):
        if word in path_graph:
            path = shortestPath(path_graph,word,pole)
            result[i] = calculate_similarity_score_using_path(path, score_graph, path_graph, threshold=threshold)
            print word,pole,result[i]
    return result

def map_word_on_axis(word, axis, score_graph=None, path_graph=None, threshold=None, model=None):
    # one pole
    if type(axis[0]) == unicode:
        if score_graph:
            return map_word_on_one_pole_axis(word, axis, score_graph, path_graph, threshold=threshold)
        else: print 'only support building vector by using graph in one pole mode'
    # two poles
    if len(axis[0]) == 2:
        if score_graph:
            return map_word_on_two_pole_axis_for_graph(word, axis, score_graph, path_graph)
        elif model:
            return map_word_on_two_pole_axis_for_model(word, axis, model)
        else: 'error'

