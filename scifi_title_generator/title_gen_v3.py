from title_tools_v2 import *
from text_tools import *

#define filenames and urls
titleFile = 'resources/titles.txt'   # List of Sci-fi titles
bodyFile= 'resources/story_text_5.txt'  # Body text we are making a title for
templateFile = 'resources/title_templates.txt'  # Templates of titles in the sci-fi style
blacklistfile = 'resources/blacklist.txt'  # words to exclude from body-text analysis


#BEGIN
##load raw texts
titles_raw = getRawWordsFromFile(titleFile)
body_raw = getRawWordsFromFile(bodyFile)

##keywords & markups
standard = makeStandardForRelevance()  # get standard counts from a standard corpus
blacklist = getBlackList(blacklistfile)  # read blacklist file

properNouns = getProperNounList(body_raw, blacklist)  # extract names from the body
keywords = getRelevanceForWordsInRaw(body_raw, standard, blacklist) # get sorted list of scored words

titleWords = choose_titlewords(keywords, titles_raw) #take most popular keywords, prioritizing those found in real titles
taggedTitleWords = tag_words(titleWords, body_raw) #tag the chosen title words with parts of speech
appendProperNouns(properNouns, titleWords, taggedTitleWords) #append the proper nouns to this list 


#load templates
templates = loadTemplates(templateFile) #load hand-made title templates
titleList = makeTitleList(taggedTitleWords, templates, body_raw) #apply all templates that can accommodate chosen title words
#print(titleList)