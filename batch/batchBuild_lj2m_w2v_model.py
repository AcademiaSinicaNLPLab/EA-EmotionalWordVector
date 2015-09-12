# -*- coding: utf8 -*-
import sys, os
sys.path.append("../")
import cPickle as pickle
from models import buildmodel

def help():
    print "usage: python [model_save_path]"
    print
    print " e.g.: python /corpus/LJ2M/exp/model/lj2mStanfordparser_300features_50minwords_10context"
    exit(-1)

if __name__ == '__main__':

    if len(sys.argv) != 2: help()
    raw_data_root = '/home/bs980201/projects/github_repo/LJ2M/raw/'
    emotions = sorted([x for x in os.listdir(raw_data_root) if not x.startswith('.')])
    w2v_sentences = []
    emotion_sentences = []

    # emotions = emotions[0:2]
    # print emotions
    for i,emotion in enumerate(emotions):
        # docs = [[[1,2,3],[4,5,6],[7,8,9]],[[1,1,1],[2,2,2],[3,3,3]]] ## for test
        print 'Start to load '+emotion+'_wordlists.pkl'
        docs = pickle.load( open('/corpus/LJ2M/data/pkl/lj2m_wordlists/'+emotion+'_wordlists.pkl', "rb" ) )
        for doc in docs:
            emotion_sentences += doc
        w2v_sentences += emotion_sentences
        del emotion_sentences
        emotion_sentences = []
        print ">> %s. %s emotion doc finishing!" % (i+1,emotion)
        del docs

    print 'Start to build the w2v model'
    num_features = 300    # Word vector dimensionality                      
    min_word_count = 50   # Minimum word count                        
    context = 10        # Context window size    
    model_save_path = sys.argv[1]             

    buildmodel.build_word_vector(w2v_sentences,model_save_path,num_features,min_word_count,context)