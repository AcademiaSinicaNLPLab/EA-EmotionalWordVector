from __future__ import generators
import sys
sys.path.append("../")
import cPickle as pickle
import numpy as np
import collections
import nltk
from nltk.corpus import wordnet as wn
from scipy import spatial
import math
from setting.category import LJ40K, LJ40K_v ,LJ40K_n, LJ40K_j, Emotion_wheel, Feeling_Wheel
from models.modeltools import txt_to_wordvecmodel

# Priority dictionary using binary heaps
# David Eppstein, UC Irvine, 8 Mar 2002
class priorityDictionary(dict):
    def __init__(self):
        '''Initialize priorityDictionary by creating binary heap
of pairs (value,key).  Note that changing or removing a dict entry will
not remove the old pair from the heap until it is found by smallest() or
until the heap is rebuilt.'''
        self.__heap = []
        dict.__init__(self)

    def smallest(self):
        '''Find smallest item after removing deleted items from heap.'''
        if len(self) == 0:
            raise IndexError, "smallest of empty priorityDictionary"
        heap = self.__heap
        while heap[0][1] not in self or self[heap[0][1]] != heap[0][0]:
            lastItem = heap.pop()
            insertionPoint = 0
            while 1:
                smallChild = 2*insertionPoint+1
                if smallChild+1 < len(heap) and \
                        heap[smallChild] > heap[smallChild+1]:
                    smallChild += 1
                if smallChild >= len(heap) or lastItem <= heap[smallChild]:
                    heap[insertionPoint] = lastItem
                    break
                heap[insertionPoint] = heap[smallChild]
                insertionPoint = smallChild
        return heap[0][1]
    
    def __iter__(self):
        '''Create destructive sorted iterator of priorityDictionary.'''
        def iterfn():
            while len(self) > 0:
                x = self.smallest()
                yield x
                del self[x]
        return iterfn()
    
    def __setitem__(self,key,val):
        '''Change value stored in dictionary and add corresponding
pair to heap.  Rebuilds the heap if the number of deleted items grows
too large, to avoid memory leakage.'''
        dict.__setitem__(self,key,val)
        heap = self.__heap
        if len(heap) > 2 * len(self):
            self.__heap = [(v,k) for k,v in self.iteritems()]
            self.__heap.sort()  # builtin sort likely faster than O(n) heapify
        else:
            newPair = (val,key)
            insertionPoint = len(heap)
            heap.append(None)
            while insertionPoint > 0 and \
                    newPair < heap[(insertionPoint-1)//2]:
                heap[insertionPoint] = heap[(insertionPoint-1)//2]
                insertionPoint = (insertionPoint-1)//2
            heap[insertionPoint] = newPair
    
    def setdefault(self,key,val):
        '''Reimplement setdefault to call our customized __setitem__.'''
        if key not in self:
            self[key] = val
        return self[key]

# Dijkstra's algorithm for shortest paths
# David Eppstein, UC Irvine, 4 April 2002
# http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/117228
# from priodict import priorityDictionary
def Dijkstra(G,start,end=None):
    """
    Find shortest paths from the start vertex to all
    vertices nearer than or equal to the end.

    The input graph G is assumed to have the following
    representation: A vertex can be any object that can
    be used as an index into a dictionary.  G is a
    dictionary, indexed by vertices.  For any vertex v,
    G[v] is itself a dictionary, indexed by the neighbors
    of v.  For any edge v->w, G[v][w] is the length of
    the edge.  This is related to the representation in
    <http://www.python.org/doc/essays/graphs.html>
    where Guido van Rossum suggests representing graphs
    as dictionaries mapping vertices to lists of neighbors,
    however dictionaries of edges have many advantages
    over lists: they can store extra information (here,
    the lengths), they support fast existence tests,
    and they allow easy modification of the graph by edge
    insertion and removal.  Such modifications are not
    needed here but are important in other graph algorithms.
    Since dictionaries obey iterator protocol, a graph
    represented as described here could be handed without
    modification to an algorithm using Guido's representation.

    Of course, G and G[v] need not be Python dict objects;
    they can be any other object that obeys dict protocol,
    for instance a wrapper in which vertices are URLs
    and a call to G[v] loads the web page and finds its links.
    
    The output is a pair (D,P) where D[v] is the distance
    from start to v and P[v] is the predecessor of v along
    the shortest path from s to v.
    
    Dijkstra's algorithm is only guaranteed to work correctly
    when all edge lengths are positive. This code does not
    verify this property for all edges (only the edges seen
    before the end vertex is reached), but will correctly
    compute shortest paths even for some graphs with negative
    edges, and will raise an exception if it discovers that
    a negative edge has caused it to make a mistake.
    """

    D = {}  # dictionary of final distances
    P = {}  # dictionary of predecessors
    Q = priorityDictionary()   # est.dist. of non-final vert.
    Q[start] = 0
    
    for v in Q:
        D[v] = Q[v]
        if v == end: break
        
        for w in G[v]:
            vwLength = D[v] + G[v][w]
            if w in D:
                if vwLength < D[w]:
                    raise ValueError, \
  "Dijkstra: found better path to already-final vertex"
            elif w not in Q or vwLength < Q[w]:
                Q[w] = vwLength
                P[w] = v
    
    return (D,P)
            
def shortestPath(G,start,end):
    """
    Find a single shortest path from the given start vertex
    to the given end vertex.
    The input has the same conventions as Dijkstra().
    The output is a list of the vertices in order along
    the shortest path.
    """

    D,P = Dijkstra(G,start,end)
    Path = []
    while 1:
        Path.append(end)
        if end == start: break
        try: end = P[end]
        except: return []
    Path.reverse()
    return Path


def get_all_synsets(word, pos=None):
    for ss in wn.synsets(word):
        for lemma in ss.lemma_names():
            yield (lemma, ss.name())

def get_all_synsets(word, pos=None):
    for ss in wn.synsets(word, pos):
        for lemma in ss.lemma_names():
            # yield (lemma, ss.name())
            yield (lemma, ss)

def get_all_similar_tos(word, pos=None):
    for ss in wn.synsets(word, pos):
            for sim in ss.similar_tos():
                for lemma in sim.lemma_names():
                    # yield (lemma, sim.name())
                    yield (lemma, sim)

def get_all_antonyms(word, pos=None):
    for ss in wn.synsets(word, pos):
        for sslema in ss.lemmas():
            for antlemma in sslema.antonyms():
                    # yield (antlemma.name(), antlemma.synset().name())
                    yield (antlemma.name(), antlemma.synset())

def get_all_hyponyms(word, pos=None):
    for ss in wn.synsets(word, pos=pos):
            for hyp in ss.hyponyms():
                for lemma in hyp.lemma_names():
                    # yield (lemma, hyp.name())
                    yield (lemma, hyp)

def get_all_also_sees(word, pos=None):
        for ss in wn.synsets(word):
            for also in ss.also_sees():
                for lemma in also.lemma_names():
                    # yield (lemma, also.name())
                    yield (lemma, also)

def get_all_synonyms_antonyms(word, rel=None, pos=None):
    if rel == None: rel = ['ss','ant','sim']
    if 'ss' in rel:
        for x in get_all_synsets(word, pos):
            yield (x[0], x[1], 'ss')
    if 'ant' in rel:
        for x in get_all_antonyms(word, pos):
            yield (x[0], x[1], 'ant')
    if 'sim' in rel:
        for x in get_all_similar_tos(word, pos):
            yield (x[0], x[1], 'sim')

def weight_between_words(word1, word2, **kwargs):
    model = kwargs['model'] if 'model' in kwargs else None
    related_word_type = kwargs['related_word_type'] if 'related_word_type' in kwargs else None
    if model:
        if related_word_type:          
            if related_word_type == 'ss' or related_word_type == 'sim':
                ## from similar to dissimilar: 1 ~ 0
                return (2.0 - spatial.distance.cosine(model[word1], model[word2]))/2.0
            elif related_word_type == 'ant':
                return -1.0
        else:
            ## from similar to dissimilar: 0 ~ 2
            return spatial.distance.cosine(model[word1], model[word2])
            # return -math.log((2.0 - spatial.distance.cosine(model[word1], model[word2]))/2.0)
    else:
        if related_word_type:
            if related_word_type == 'ss' or related_word_type == 'sim':
                return 1.0
            elif related_word_type == 'ant':
                return -1.0
        else: return 1.0

def graph_for_calculate_score(word, related_word, related_word_type, graph, model=None):    
        if word not in graph[word]:
            graph[word][word] = 1.0
        if related_word not in graph[related_word]:
            graph[related_word][related_word] = 1.0

        if related_word not in graph[word]:
            if model:
                ## e.g. edge weight: 1, -1, 0.8
                graph[word][related_word] = weight_between_words(word, related_word, related_word_type=related_word_type, model=model)
            else:
                ## e.g. edge weight: 1, -1
                graph[word][related_word] = weight_between_words(word, related_word, related_word_type=related_word_type)
        if word not in graph[related_word]:
            if model:
                ## e.g. edge weight: 1, -1, 0.8
                graph[related_word][word] = weight_between_words(related_word, word, related_word_type=related_word_type, model=model)
            else:
                ## e.g. edge weight: 1, -1
                graph[related_word][word] = weight_between_words(word, related_word, related_word_type=related_word_type)


def graph_for_calculate_path(word, related_word, graph, model=None):
    if related_word not in graph[word] and related_word != word:
        if model:
            ## e.g. edge weight: 0.8, 1.2
            graph[word][related_word] = weight_between_words(word, related_word, model=model)
        else:
            ## e.g. edge weight: all 1
            graph[related_word][word] =  weight_between_words(word, related_word)
    if word not in graph[related_word] and related_word != word:
        if model:
            ## e.g. edge weight: 0.8, 1.2
            graph[related_word][word] = weight_between_words(related_word, word, model=model)
        else:
            ## e.g. edge weight: all 1
            graph[related_word][word] = weight_between_words(related_word, word)

def pos_filter(word, wordlist, pos):
    '''
    if word in wordlist, we only concern specific pos
    e.g. 'excited' in LJ40K, we only link words which are adj to 'excited'
    '''
    if word in wordlist: return pos
    else: return [None]

def build_graph(data, mode, graph=None, rel=None, pos_list=None, modelpath=None):
    '''
    4 usages:
        graph,wordset = build_graph(data,'calculate_path', rel=['ss','ant','sim'], modelpath='../data/model/glove.42B.300d.txt')
        graph,wordset = build_graph(data,'calculate_path', rel=['ss','ant','sim'])
        graph,wordset = build_graph(data,'calculate_score', rel=['ss','ant','sim'], modelpath='../data/model/glove.42B.300d.txt')
        graph,wordset = build_graph(data,'calculate_score', rel=['ss','ant','sim'])
    '''

    word_set = set()
    graph = graph if graph else collections.defaultdict(dict)
    wnl = nltk.WordNetLemmatizer()

    if modelpath:
        model = txt_to_wordvecmodel(modelpath)
        model_wordset = set(model.keys())
    else:
        model = None

    for i, word in enumerate(data):
        #225534 words in data
        if i % 10000 == 0: print i
        if type(word) == str:
            word = word.decode('utf-8')
        word = word.lower()
        word_exist_in_wordnet = wn.morphy(word) ### bring lots of noises
        if word_exist_in_wordnet:
            word = wnl.lemmatize(word)
            word_set.add(word)
            ## 35304 remain

            pos_list = pos_filter(word, LJ40K, ['s','a'])
            if pos_list == None: pos_list=[None]
            for i,p in enumerate(pos_list):
                related_words = get_all_synonyms_antonyms(word, rel, p)
                rws = list(related_words)
                if rws:
                    for w in rws:
                        related_word = w[0].lower()
                        related_word_type = w[2]
                        if model:
                            if related_word not in model_wordset or word not in model_wordset: continue
                            if mode == 'calculate_path':
                                graph_for_calculate_path(word, related_word, graph, model)
                            elif mode == 'calculate_score':
                                graph_for_calculate_score(word, related_word, related_word_type, graph, model)
                        else:
                            if mode == 'calculate_path':
                                graph_for_calculate_path(word, related_word, graph)
                            elif mode == 'calculate_score':
                                graph_for_calculate_score(word, related_word, related_word_type, graph)
    return graph, word_set


if __name__ == '__main__':
    data = pickle.load(open('/corpus/LJ40K/data/features/tfidf/GlobalInfo.pkl'))
    
    graph,wordset = build_graph(data,'calculate_path', rel=['ss','ant','sim'], modelpath='../data/model/glove.42B.300d.txt')
    # graph,wordset = build_graph(data,'calculate_path', rel=['ss','ant','sim'])
    # graph,wordset = build_graph(data,'calculate_score', rel=['ss','ant','sim'], modelpath='../data/model/glove.42B.300d.txt')
    # graph,wordset = build_graph(data,'calculate_score', rel=['ss','ant','sim'])
    
    # print len(wordset)
    print len(graph)
    # pickle.dump(graph,open('graph.pkl', 'wb'))