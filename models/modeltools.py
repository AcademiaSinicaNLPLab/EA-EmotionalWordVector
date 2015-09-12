# -*- coding: utf8 -*-
import sys, os
sys.path.append("../")

def txt_to_wordvecmodel(filepath):
    model = {}
    c = 0
    for line in open(filepath, 'r'):
        c = c + 1
        if c % 5000 == 0: print c
        wordlist = line.strip().split()
        key = wordlist[0].decode('utf-8')
        wordlist = [float(e) for e in wordlist[1:]]
        model[key] = wordlist
    return model

def wordvecmodel_to_txt(model,savepath):
    file = open(savepath, 'w')
    for word in model:
        if type(model[word]) != list:
            wordvec = model[word].tolist()
        else: wordvec = model[word]
        assert type(wordvec) == list
        wordvec = ' '.join(str(x) for x in wordvec)
        line = word+' '+wordvec+'\n'
        file.write(line)
    file.close()

def wordvecmodel_filter(model,required_wordset):
    new_model = {w:model[w] for w in model if w in required_wordset}
    return new_model

def find_the_most_similar_words_of_a_wordvec(word_vector, worddict):
    most_similar = []
    for w in worddict:
        value = (2.0 - spatial.distance.cosine(worddict[w], word_vector))/2.0
        most_similar.append((value, w))

    ms_s = sorted(most_similar)
    return ms_s[-50:]

def find_the_most_similar_words(word, worddict):
    most_similar = []
    for w in worddict:
        value = (2.0 - spatial.distance.cosine(worddict[w], worddict[word]))/2.0
        most_similar.append((value, w, worddict[w]))

    ms_s = sorted(most_similar)
    return ms_s[-50:]