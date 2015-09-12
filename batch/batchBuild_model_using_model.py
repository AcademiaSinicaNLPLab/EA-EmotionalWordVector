# -*- coding: utf8 -*-
import cPickle as pickle
import sys, os
sys.path.append('../')
from gensim.models import Word2Vec
from models.buildmodel import map_word_on_axis, build_axis
from setting.category import LJ40K, Feeling_Wheel

if __name__ == '__main__':
    ## for LJ2M word vector model
    w2vmodel = Word2Vec.load('/corpus/LJ2M/exp/model/lj2mStanfordparser_300features_50minwords_10context')
    index2word_set = set(w2vmodel.index2word)

    # # for 42B word vector model
    # w2vmodel = txt_to_wordvecmodel(filepath='glove.42B.300d.txt')
    # index2word_set = set(w2vmodel.keys())

    use_unicode = False
    if type(list(index2word_set)[0]) == unicode:
        use_unicode = True


    model = {}
    data = pickle.load(open('/corpus/LJ40K/data/features/tfidf/GlobalInfo.pkl'))
    LJ40K_axis = build_axis(LJ40K)

    for i,word in enumerate(data):
        if i % 1000 == 0: print i
        if use_unicode:
            word = word.decode('utf-8')
        if word in index2word_set:
            # case1:
            model[word] = map_word_on_axis(word, Feeling_Wheel, model=w2vmodel)
            # case2:
            # model[word] = map_word_on_axis(word, LJ40K_axis, model=w2vmodel)

    print 'dump..'
    pickle.dump(model,open('model_wordvec_semantic_similarity_lemma_63768_LJ2M_Feeling_Wheel.pkl', 'wb'))