import nltk
import spacy
from nltk.corpus import stopwords
nlp = spacy.load('en')

def printSentInfo(sent_nlp):
    tags = [(w.text,  w.tag_, w.pos_, w.dep_) for w in sent_nlp]
    for t in tags:
        print(t)

def getRawWordsFromFile(filename):
    f = open(filename, 'rU', encoding='utf8')
    raw = f.read()
    f.close()
    return raw

def get_scrubbed_words(raw, removeProps = True):
    swapped = raw.replace('\n', '. ')
    if removeProps:
        process = nlp(swapped)
        words = [w.text.lower() for w in process if w.pos_!='PROPN']
    else:
        process = swapped.split()
        words = [w.lower() for w in process]
    #words = nltk.word_tokenize(swapped.lower())
    stops = stopwords.words('english')
    scrubbed = [token for token in words if token not in stops and token.isalnum()]
    return scrubbed

def makeSentsFromRaw(raw):
    #raw = raw.replace('\n\n', '\n ')
    swapped = raw.replace('\n', ' . ')
    #swapped = swapped.replace('..', '. ')
    sents = nltk.sent_tokenize(swapped)
    sents = [s for s in sents if s!='.']
    return sents