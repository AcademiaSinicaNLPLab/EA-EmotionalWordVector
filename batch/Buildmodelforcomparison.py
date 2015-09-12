# -*- coding: utf8 -*-
import sys, os
sys.path.append("../")
from models.modeltools import txt_to_wordvecmodel, wordvecmodel_to_txt, wordvecmodel_filter
import cPickle as pickle

'''
input:
model1, model2, filter_set, filepath1, filepath2
'''
if __name__ == '__main__':

    model1 = txt_to_wordvecmodel(filepath='../data/model/glove.42B.300d.txt')
    model2 = pickle.load(open('../data/model/model_wordvec_semantic_similarity_lemma_35304_42b_Feeling_Wheel.pkl'))
    filter_set = pickle.load(open('../data/wordset/wordsetlemma_basickeyword_LJ40K_FeelingWheel_1122.pkl'))

    model1 = wordvecmodel_filter(model1, filter_set)
    model2 = wordvecmodel_filter(model2, filter_set)

    filter_set = set(model1.keys()) & set(model2.keys())

    model1 = wordvecmodel_filter(model1, filter_set)
    model2 = wordvecmodel_filter(model2, filter_set)
    

    print 'length of the filtered model1 is ',len(model1)
    print 'length of the filtered model2 is ',len(model2)

    filepath1 = '../textSNE/testdata/w2v_worddict_42B_1122.txt'
    filepath2 = '../textSNE/testdata/model_wordvec_semantic_similarity_lemma_35304_42b_Feeling_Wheel_1122.txt'

    wordvecmodel_to_txt(model1,filepath1)
    os.system('gzip '+filepath1)
    wordvecmodel_to_txt(model1,filepath1)

    wordvecmodel_to_txt(model2,filepath2)
    os.system('gzip '+filepath2)
    wordvecmodel_to_txt(model2,filepath2)