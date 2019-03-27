import collections
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from FunctionsKfolds import getData
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import SnowballStemmer
from TrainModelW2V import preprocessGensim
from IndividualFunctions import recoverySentences
stemmer = SnowballStemmer('spanish')
toktok = ToktokTokenizer()


def clusterSentences(sentences, nb_of_clusters):
        #tfidfVectorizer = TfidfVectorizer()
        tfidfVectorizer = TfidfVectorizer(tokenizer=preprocessGensim,lowercase=False)
        #tfidfVectorizer = TfidfVectorizer(lowercase=False)
        #builds a tf-idf matrix for the sentences
        tfidfMatrix = tfidfVectorizer.fit_transform(sentences)
        kmeans = KMeans(n_clusters=nb_of_clusters)
        kmeans.fit(tfidfMatrix)
        clusters = collections.defaultdict(list)
        for i, label in enumerate(kmeans.labels_):
                clusters[label].append(i)
        return dict(clusters)

def concatenanteVector(vectorSentences):
    my_file = open("Clustering/textClustering.txt", "a", encoding='utf-8')
    for i in range(len(vectorSentences)):
        my_file.write(vectorSentences[i])
    my_file.write("\n")
    my_file.close()

if __name__ == "__main__":
    nclusters= 80
    trainSentences = getData('TrainData/')
    #classSentencesPreprocess = list(filter(None, [preprocessGensim(trainSentences[i]) for i in range(len(trainSentences))]))
    classSentencesPreprocess = [preprocessGensim(trainSentences[i]) for i in range(len(trainSentences))]
    corpusClass = np.unique(classSentencesPreprocess, return_index = True)
    classSentencesRecovery = recoverySentences(corpusClass[1], trainSentences)
    np.savetxt("Clustering/corpusClass.txt", corpusClass, delimiter=',', newline = "", fmt="%s", encoding='utf-8')
    np.savetxt("Clustering/classSentencesRecovery.txt", classSentencesRecovery, delimiter=',', newline = "", fmt="%s", encoding='utf-8')
    corpusClass_1 = np.unique(classSentencesRecovery, return_index = True)

    clusters = clusterSentences(classSentencesRecovery, nclusters)
    matrixCluster=[[(classSentencesRecovery[sentence]) for i,sentence in enumerate(clusters[cluster])] for cluster in range(nclusters)]
    open('Clustering/textClustering.txt', 'w').close()
    for i in range(len(matrixCluster)):
        concatenanteVector(matrixCluster[i])
    
    print('Text Clustering Finished')
