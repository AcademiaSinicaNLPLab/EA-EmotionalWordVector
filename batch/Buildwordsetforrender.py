import cPickle as pickle
import pymongo
from pymongo import MongoClient
from setting.category import LJ40K, Feeling_Wheel

client = MongoClient('doraemon.iis.sinica.edu.tw:27017')
LJ40Kdb = client.LJ40K
keyword_source = LJ40Kdb['resource.WordNetAffect']
basic_keyword_list = [ mdoc['word'].encode('utf-8') for mdoc in list(keyword_source.find({'type':'basic'}))]
basic_keyword_list = [w.decode('utf-8') for w in basic_keyword_list]


basic_keyword_set = set(basic_keyword_list)
LJ40Kset = set(LJ40K)
feelingwheel_set = set()
for e1,e2 in Feeling_Wheel:
    feelingwheel_set.add(e1)
    feelingwheel_set.add(e2)


wanted_set = basic_keyword_set|LJ40Kset|feelingwheel_set

ws = set()
for w in wanted_set:
    ws.add(w.encode('utf-8'))

pickle.dump(ws,open('wordset_basickeyword_LJ40K_FeelingWheel.pkl', 'wb'))