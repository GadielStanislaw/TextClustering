import time
start = time.time()
import numpy as np
from TrainModelW2V import getDataFiles, filesJoin, preprocessGensim, trainWord2Vec
if __name__ == "__main__":
    
    #Read of data for train an add the files for obtain model W2V
    filesSentences = getDataFiles('TestFile2/')
    print('\n Archivos Cargados ...\n')
    fileSentences= filesJoin(filesSentences)
    initialData=[fileSentences[i] for i in range(len(fileSentences))]
    print('\n Archivos Unidos ...\n')
    preprocessCorpus = list(filter(None, [preprocessGensim(initialData[i]) for i in range(len(initialData))]))
    print('\n Sentencias Preprocesadas ...\n')
    corpusData = np.unique(preprocessCorpus)
    np.savetxt("corpusData.txt", corpusData, newline = "\n", fmt="%s", encoding='utf-8')
    print('\n Iniciando carga de Datos Preprocesados ...\n')
    trainWord2Vec(corpusData,'modelClass.bin')
    end = time.time()
    print('Time :', end - start,'sec')