import spacy
import nltk
from nltk.corpus import stopwords
from nltk.corpus import brown
import re

import collections
from math import log
import random
import itertools

from text_tools import *

nlp = spacy.load('en')
english_vocab = set(w.lower() for w in nltk.corpus.words.words())


def getBlackList(file):
    raw = getRawWordsFromFile(file)
    black = raw.split('\n')
    black = [b for b in black if b != '']
    return black

def getProperNounList(raw, blacklist):
    excludelist = blacklist[:]
    excludelist.append("’s") #exclude possessives
    proper = collections.Counter()
    sents = makeSentsFromRaw(raw)

    for sent in sents:
        sent_nlp = nlp(sent)
        pos = [w.pos_ for w in sent_nlp]
        running = ''
        started = False
        for i in range(0, len(sent_nlp)):
            w = sent_nlp[i].text.lower()
            if pos[i]=='PROPN' and w not in excludelist:
                running += w + ' '
                started = True
            elif started and sent_nlp[i].tag_=='HYPH':
                running = running[0:-1]
                running += w
            elif started == True:
                #sanitize to trim
                running = running.strip()
                proper.update([running])
                started = False
                running = ''

    return proper

def makeStandardForRelevance():
    brown_news_articles = [' '.join(brown.words(fileids=fileid)) for fileid in brown.fileids(['news'])]
    brown_token_sets = [get_scrubbed_words(raw_article) for raw_article in brown_news_articles]
    brown_counts = [collections.Counter(tokens) for tokens in brown_token_sets]
    return brown_counts

def getRelevanceForWordsInRaw(body, standard, blacklist):

    body_tokens = get_scrubbed_words(body)
    body_count = collections.Counter(body_tokens)
    standard.append(body_count)

    idf = {}
    tfidf = {}
    NCorpa = len(standard)
    for token in body_count:
        if token in english_vocab and token not in blacklist:
            idf[token] = log(NCorpa/len([doc_index for doc_index in range(NCorpa) if standard[doc_index][token]>0]))
            tfidf[token] = body_count[token]*idf[token]

    terms_sorted = sorted(tfidf.items(), key=lambda x: -x[1])

    return terms_sorted


def choose_titlewords(keywords, titles_raw, maxWords = 3, bonusThreshold = 3):
    #get "bonus words" that are popular in real titles. These words are prioritized.
    titles =  makeSentsFromRaw(titles_raw)
    tokenized = [nltk.word_tokenize(t.lower()) for t in titles]
    stops = stopwords.words('english')
    scifi_words_counter = collections.Counter()
    for t in tokenized:
        for w in t:
            scifi_words_counter.update([w])
    bonus_words = [w.lower() for w in scifi_words_counter if
                   scifi_words_counter[w] >= bonusThreshold and w.lower() not in stops]


    titlewords = [key for key in keywords if key in bonus_words]
    titlewords = titlewords[:maxWords]

    #add proper nouns if in special list

    Nadd = maxWords - len(titlewords)
    additional = [key for key in keywords if (key not in titlewords)]
    titlewords_ex = titlewords + additional[:Nadd]
    titlewords = [wp[0] for wp in titlewords_ex]

    return titlewords





def get_tags_for_word(sent_nlp, index, annotations):
    tuples = []
    inlength = len(annotations)
    for depth in range(0,3):
        if index > depth-1:
            tuples.append((sent_nlp[index - depth].pos_, sent_nlp[index - depth].dep_, sent_nlp[index-depth].tag_, sent_nlp[index-depth].text))
        else:
            tuples.append(('', '', '', ''))

    def joinup(tt, n):
        words=[]
        for i in range(2,-1,-1):
            words.append(tt[i][3])
        return ' '.join(words)

    def test(teststrs, testslot, annotations, newTag):
        ntests = len(teststrs)
        for i in range(0,ntests):
            if tuples[i][testslot] != teststrs[-(1+i)]:
                return False
        annotations[newTag].append(joinup(tuples,ntests))
        return True

    ### APPLY RULES -- do not test for propper nouns
    test(['nsubj'], 1, annotations, 'SUBJ')
    test(['prep','pobj'], 1, annotations, 'PREP OBJ')
    test(['prep', 'det', 'pobj'], 1, annotations, 'PREP DT OBJ')
    test(['prep','amod','pobj'], 1, annotations, 'PREP ADJ OBJ')
    test(['amod', 'pobj'], 1, annotations, 'ADJ OBJ')
    test(['prep','compound','pobj'],1,annotations, 'PREP COMP OBJ')
    test(['compound', 'nsubj'], 1, annotations, 'COMP SUBJ')
    test(['poss'], 1, annotations, 'POSS')
    if test(['VBG'], 2, annotations, 'GER') == False:
        test(['VERB'], 0, annotations, 'VERB')

    test(['compound'], 1, annotations, 'COMP')
    test(['dobj'], 1, annotations, 'DOBJ')

    #if len(annotations) == inlength:
        #print(sent_nlp)
        #printSentInfo(sent_nlp)

    return annotations



def tag_words(words, body):
    sentences = makeSentsFromRaw(body)
    #for each word, find its part of speach and context
    library = {}
    for word in words:
        annotations = collections.defaultdict(list)
        for sent in sentences:
            #sent = sent.lower()
            sent_nlp = nlp(sent)
            sent_nlp_lemmas = [w.lemma_ for w in sent_nlp]
            word_lemma = nlp(word)[0].lemma_
            if word_lemma in sent_nlp_lemmas:
                index = sent_nlp_lemmas.index(word_lemma)
                annotations = get_tags_for_word(sent_nlp, index, annotations)
            elif word in sent_nlp_lemmas:
                index = sent_nlp_lemmas.index(word)
                annotations = get_tags_for_word(sent_nlp, index, annotations)

        library[word] = dict(annotations)

    return library


def appendProperNouns(properNouns, titleWords, taggedTitleWords):
    limit = 4
    n=properNouns.most_common(1)
    if n and n[0][1]>=limit:
        name = n[0][0]
        titleWords.append(name)
        tag = {'PROPN' : [name]}
        taggedTitleWords[name] = tag


def enumerate_tags(tags):
    quant = {}
    for w in tags:
        for t in tags[w]:
            quant[t] = len(tags[w][t])
    return quant


progDT = re.compile(r"<(DT)>")
progDTO = re.compile(r"<(DTO)>")
def loadTemplates(filename):
    titleTemplates=[]
    #make a regex tag finder
    prog = re.compile(r"<([\w\s]+)>")

    f = open(filename)
    for line in f.readlines():
        if '#' in line: continue
        line = line.strip('\n')
        # do special replacements for DTO and DT
        line = progDT.sub('DT', line)
        line = progDTO.sub('DTO', line)

        rhs = prog.findall(line)
        lhs = prog.sub('*', line)

        pair = (lhs, rhs)
        titleTemplates.append(pair)
    f.close()

    return titleTemplates


def getDeterminator(word, optional):
    res = 'the'
    if optional:
        if random.random()<0.5: res = ''
    return res

def titleCase(title):
    title = title.lower()
    skiplist = ['to', 'for', 'a', 'and', 'an', 'in', 'of', 'the']
    splitter = re.compile(r"([\w?|']+-?\s?)") #chop up words preserving apostraphes but not hyphens
    tsplit = splitter.split(title)
    tsplit = [t for t in tsplit if t]
    def capWord(word, skips = []):
        if word.strip() in skips:
            return word
        return word[:1].upper() + word[1:]

    caps = [capWord(t, skiplist) for t in tsplit]
    caps[0] = capWord(caps[0])
    return ''.join(caps)


starfinder = re.compile(r"(\*)")
def doTitleTemplateSub(wordset, taggedWords, template, body_raw):
    title = ''
    ntags = len(template[1])
    lhs = template[0].split('*')
    lastchunk = ''

    for i in range(0, ntags):
        title += lhs[i]
        forms = collections.Counter()
        tag = template[1][i]
        word = wordset[i]
        if tag not in taggedWords[word]:
            title = ''
            break
        contexts = taggedWords[word][tag]
        taglen = len(tag.split(' '))
        truncated = [ ' '.join(c.split()[-taglen:]) for c in contexts]
        beginsWPrep = tag.startswith('PREP') #confirm that preposition
        for t in truncated:
            okay = True
            if beginsWPrep:
                leadingword = lastchunk.split()[-1]
                phrasetofind = ' '.join([leadingword, t.split()[0]])
                okay = phrasetofind in body_raw
            if okay:
                forms.update([t])
        if len(forms)<1:
            title = ''
            break
        swap = forms.most_common()[0][0]
        lastchunk = swap
        title += swap

    if title:
        title+= lhs[ntags]
        determ = getDeterminator('', False)
        title = re.sub(r"(DT\b)", determ, title)
        determ = getDeterminator('', True)
        title = re.sub(r"(DTO)", determ, title)

    if title:
        title = titleCase(title.strip())

    return title

def makeTitleFromTemplate(taggedWords, template, body_raw):
    titles=[]
    keys = list(taggedWords)
    ntags = len(template[1])
    keyCombs = list(itertools.permutations(keys, ntags))
    for wordset in keyCombs:
        newtitle = doTitleTemplateSub(wordset, taggedWords, template, body_raw)
        if newtitle:
            titles.append(newtitle)

    return titles


def makeTitleList(taggedWords, templates, body_raw):
    titles = []
    allTags = enumerate_tags(taggedWords)
    for template in templates:
        #make sure all the parts are available
        #note: this works as long as there are no repeats in the template
        print(template)
        notValid = False
        for tag in template[1]:
            if tag not in allTags:
                notValid = True
        if notValid:
            continue
        for t in makeTitleFromTemplate(taggedWords, template, body_raw):
            print(t)
            titles.append(t)
    return titles



