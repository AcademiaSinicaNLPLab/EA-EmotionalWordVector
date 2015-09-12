# -*- coding: utf8 -*-
import cPickle as pickle
import sys, os
sys.path.append('../')
from models.buildmodel import map_word_on_axis
from setting.category import LJ40K, LJ40K_v ,LJ40K_n, LJ40K_j, Emotion_wheel, Feeling_Wheel

if __name__ == '__main__':
    '''
    only for using graph to build the model
    '''
    ## one pole
    path_graph = pickle.load(open('../data/graph/calculate_path/ss_ant_for_path_wv_LJ40k_enhance_lemma_42b_50785.pkl'))
    score_graph = pickle.load(open('../data/graph/calculate_score/pos_neg_wv_LJ40k_enhance_lemma_42b_50785.pkl'))
    words_set = pickle.load(open('../data/wordset/wordsetlemma_35304.pkl'))

    ## two pole
    # path_graph = pickle.load(open('../data/graph/calculate_path/ss_for_path_wv_lemma_42b_50172.pkl'))
    # score_graph = pickle.load(open('../data/graph/calculate_score/pos_wv_lemma_42b_50172.pkl'))
    # words_set = pickle.load(open('../data/wordset/wordsetlemma_35304.pkl'))

    model = {}
    for i,word in enumerate(words_set):
        print i
        ## one pole
        
        ##case1
        model[word] = map_word_on_axis(word, LJ40K, score_graph=score_graph, path_graph=path_graph)
        
        ##case2
        # model[word] = map_word_on_axis(word, LJ40K, score_graph=score_graph, path_graph=path_graph, threshold=0.6)

        ##case3
        # model[word] = map_word_on_axis(word, Emotion_wheel, score_graph=score_graph, path_graph=path_graph)

        ## two pole
        # model[word] = map_word_on_axis(word, Feeling_Wheel, score_graph=score_graph, path_graph=path_graph)

    print 'dump..'
    pickle.dump(model,open('model_one_minusone_semantic_similarity_lj40k_enhance_lemma_50785_42b.pkl', 'wb'))