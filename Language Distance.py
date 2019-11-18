from nltk.corpus import udhr
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy_indexed as npi
from scipy.spatial.distance import pdist,squareform
from scipy.stats import entropy
from numpy.linalg import norm
import numpy as np
from numpy import where
from sklearn.externals import joblib

 
encoding = r"-([^-]+)$"
triplets = set([])
 
encodings = []
for f in udhr.fileids():
     lang =  re.sub(encoding,"",f)
     enco = re.findall(encoding,f)
     if len(enco)>0:
         triplets |= set([(w, lang, enco[0])for w in udhr.words(f)])
words, langs, encos = zip(*triplets)
 
words,langs,encos = zip(*[t for t in triplets\
                           if t[2] in [u"UTF8",u"Latin1",u"Latin2"]])
     
words_train, words_test, langs_train, langs_test = train_test_split(
                 words, langs, test_size = 0.5, random_state = 48)
 
 
#==============================================================================
# 
# 
# nnet = MLPClassifier(hidden_layer_sizes=(120,),
#                      learning_rate_init=.001,
#                      activation='tanh',
#                      max_iter=25,
#                      alpha=1e-3,
#                      solver='adam',
#                      verbose=True,
#                      tol=1e-4,
#                      random_state=1)
# 
# 
# 
# 
# 
# word_clf = Pipeline([('vect',CountVectorizer(analyzer='char', ngram_range=(1,4))),
#                     ('tfidf',TfidfTransformer()),
#                     ('clf',nnet)])
# 
# word_clf.fit(words_train,langs_train)
#==============================================================================

#word_clf.score(words_test,langs_test)
#Gets 0.37089011535141642 score after doing several Randomized searches


##############################
#Random Search nad Export
#############################
'''
from sklearn.grid_search import RandomizedSearchCV

parameters = {'clf__learning_rate_init': (.0001,.001),
              'clf__hidden_layer_sizes':((75,),(85,),(100,)),
              'clf__alpha':(1e-4,1e-3,1e-5),
              'clf__activation': ("tanh","relu")}


rs = RandomizedSearchCV(word_clf, param_distributions=parameters,
                        n_jobs=-1, n_iter=10)

rs.fit(words_train,langs_train)

for param_name in parameters.keys():
    print("%s: %r" % (param_name, rs.best_params_[param_name]))




joblib.dump(word_clf,'model_clf.pkl')
'''

word_clf = joblib.load('model_clf.pkl')







pred_prob = word_clf.predict_proba(words_test)
avgs = npi.group_by(langs_test).mean(pred_prob)
print(avgs[1].shape)


def JSD(P,Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 *  (_P + _Q)
    return 1 - 0.5 * (entropy(_P,_M) + entropy(_Q, _M))

similarities = squareform(pdist(avgs[1], metric =JSD))

ixdutch = where(word_clf.classes_ == u"Dutch_Nederlands")[0][0]
ixenglish = where(word_clf.classes_ == u"English")[0][0]
ixafrikaans = where(word_clf.classes_ == u"Afrikaans")[0][0]
ixfrench = where(word_clf.classes_ == u"French_Francais")[0][0]
ixhungarian = where(word_clf.classes_ == u"Hungarian_Magyar")[0][0]
ixlatvian = where(word_clf.classes_ == u"Latvian")[0][0]
ixgerman = where(word_clf.classes_ == u"German_Deutsch")[0][0]
ixfijian = where(word_clf.classes_ == u"Fijian")[0][0]
ixmaori = where(word_clf.classes_ == u"Maori")[0][0]

#print ("Dutch-Afrikaans:\t",similarities[ixdutch,ixafrikaans])
#print ("Dutch-German:\t\t",similarities[ixdutch,ixgerman])
#print ("Dutch-English:\t\t",similarities[ixdutch,ixenglish])
#print ("Dutch-French:\t\t",similarities[ixdutch,ixfrench])
#print ("Dutch-Latvian:\t\t",similarities[ixdutch,ixlatvian])
#print ("Dutch-Hungarian:\t",similarities[ixdutch,ixhungarian])
#print ("Dutch-Fijian:\t\t",similarities[ixdutch,ixfijian])
print ("Maori-Fijian:\t\t",similarities[ixmaori,ixfijian])


ixlatin = where(word_clf.classes_ == u"Latin_Latina")[0][0]
ixspanish = where(word_clf.classes_ == u"Spanish_Espanol")[0][0]
ixgalego = where(word_clf.classes_ == u"Galician_Galego")[0][0]
ixausturian = where(word_clf.classes_ == u"Asturian_Bable")[0][0]
ixportugues = where(word_clf.classes_ == u"Portuguese_Portugues")[0][0]
ixitalian = where(word_clf.classes_ == u"Italian_Italiano")[0][0]
ixpolish = where(word_clf.classes_ == u"Polish_Polski")[0][0]
ixrussian = where(word_clf.classes_ == u"Russian_Russky")[0][0]
ixturkish = where(word_clf.classes_ == u"Turkish_Turkce")[0][0]
ixromanian = where(word_clf.classes_ == u"Romanian_Romana")[0][0]
ixzapoteco = where(word_clf.classes_ == u"Zapoteco")[0][0]
ixhebrew = where(word_clf.classes_ == u"Hebrew_Ivrit")[0][0]
ixsardinian = where(word_clf.classes_ == u"Sardinian")[0][0]
ixbasque = where(word_clf.classes_ == u"Basque_Euskara")[0][0]
ixcatalan = where(word_clf.classes_ == u"Catalan_Catala")[0][0]
ixnahuatl = where(word_clf.classes_ == u"Nahuatl")[0][0]
ixmaya = where(word_clf.classes_ == u"Mayan_Yucateco")[0][0]
ixmixteco = where(word_clf.classes_ == u"Mixteco")[0][0]
ixotomi = where(word_clf.classes_ == u"Otomi_Nahnu")[0][0]
ixtotonaco = where(word_clf.classes_ == u"Totonaco")[0][0]
ixmazateco = where(word_clf.classes_ == u"Mazateco")[0][0]
ixtzeltal = where(word_clf.classes_ == u"Tzeltal")[0][0]
ixtzotzil = where(word_clf.classes_ == u"Tzotzil")[0][0]


print("Languages compared to Latin:\nRomance Languages share a higher simularity.\
      But... there's not a great difference between Native Languages to Mexico")
print ("Latin-Spanish:\t\t",similarities[ixlatin,ixspanish])
print ("Latin-French:\t\t",similarities[ixlatin,ixfrench])
print ("Latin-Italian:\t\t",similarities[ixlatin,ixitalian])
print ("Latin-English:\t\t",similarities[ixlatin,ixenglish])
print ("Latin-Sardin.:\t\t",similarities[ixlatin,ixsardinian])
print ("Latin-romanian:\t\t",similarities[ixlatin,ixromanian])
print ("Latin-English:\t\t",similarities[ixlatin,ixausturian])
print ("Latin-Portu.:\t\t",similarities[ixlatin,ixportugues])
print ("Latin-Turkish:\t\t",similarities[ixlatin,ixturkish])
print ("Latin-Polish:\t\t",similarities[ixlatin,ixpolish])
print ("Latin-Hebrew:\t\t",similarities[ixlatin,ixhebrew])
print ("Latin-Basque:\t\t",similarities[ixlatin,ixbasque])
print ("Latin-Catalan:\t\t",similarities[ixlatin,ixcatalan])
print ("Latin-German:\t\t",similarities[ixlatin,ixgerman])
print ("Latin-Mixteco:\t\t",similarities[ixlatin,ixmixteco])
print ("Latin-Otomi:\t\t",similarities[ixlatin,ixotomi])
print ("Latin-Maya:\t\t",similarities[ixlatin,ixmaya])

print("Maya to other Languge Comparisons:")
print ("Maya-Spanish:\t\t",similarities[ixmaya,ixspanish])
print ("Maya-Nahuatl:\t\t",similarities[ixmaya,ixnahuatl])
print ("Maya-English:\t\t",similarities[ixmaya,ixenglish])
print ("Maya-French:\t\t",similarities[ixmaya,ixfrench])
print ("Maya-Mixteco:\t\t",similarities[ixmaya,ixmixteco])
print ("Maya-Otomi:\t\t",similarities[ixmaya,ixotomi])
print ("Maya-Totonaco:\t\t",similarities[ixmaya,ixtotonaco])
print ("Maya-Mazateco:\t\t",similarities[ixmaya,ixmazateco])
print ("Maya-Tzeltal:\t\t",similarities[ixmaya,ixtzeltal])
print ("Maya-Tzotzil:\t\t",similarities[ixmaya,ixtzotzil])

print("Surpisingly Maya is closer to English, than Spanish.")
print("Maya is certainly closer to other indigienous languages,\n\
      and it becomes clear that Tzeltal and Tzotzil are closely related\n\
      and indeed are Mayan languages.")
