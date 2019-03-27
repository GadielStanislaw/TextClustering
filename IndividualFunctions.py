import nltk
from FunctionsKfolds import lastElement
def recoverySentences(preprocessSentences,initialSentences):
    return [initialSentences[preprocessSentences[i]] for i in range(len(preprocessSentences))]

def frecuenceSentences (sentencesPredict):
    sentencesLastElement= [lastElement(sentencesPredict[i]) for i in range(len(sentencesPredict))]
    frequenceClass=nltk.FreqDist(sentencesLastElement)
    sortedFrequence=sorted(frequenceClass.items(), key = lambda item: item[1], reverse=True)
    l=len(sortedFrequence)
    if l > 3:
        return sortedFrequence[0], sortedFrequence[1], sortedFrequence[2]
    elif l <=3:
        return sortedFrequence