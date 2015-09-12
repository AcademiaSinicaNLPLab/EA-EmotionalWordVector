# -*- coding: utf8 -*-
import sys, os
sys.path.append("../")
import cPickle as pickle
from graph.buildgraph import build_graph

if __name__ == '__main__':
    data = pickle.load(open('/corpus/LJ40K/data/features/tfidf/GlobalInfo.pkl'))
 
    graph,wordset = build_graph(data,'calculate_path', rel=['ss','ant','sim'], pos_list=['v'] ,modelpath='../data/model/glove.42B.300d.txt')
    
    print len(wordset)
    print len(graph)

    pickle.dump(graph,open('../data/graph/calculate_path/test.pkl', 'wb'))