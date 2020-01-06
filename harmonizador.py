import glob
import ntpath
import numpy as np
import csv
import os
from keras.models import model_from_json
from keras_preprocessing import sequence

def carregamento():
    # carrega os modelos existentes
    pasta_modelo = 'json_modelo/'
    arquivos_modelo = os.listdir(pasta_modelo)
    # lista de modelos disponiveis
    for indice, arquivo in enumerate(arquivos_modelo):
        print(str(indice) + ": " + arquivo)
    # selecao do modelo
    escolha_modelo = int(input('Escolha o modelo: '))
    arquivo_modelo = arquivos_modelo[escolha_modelo]
    caminho_modelo = '%s%s' % (pasta_modelo, arquivo_modelo)

    # carrega os pesos de treino disponiveis
    pasta_pesos = 'pesos_modelo/'
    arquivos_pesos = os.listdir(pasta_pesos)
    for indice, arquivo in enumerate(arquivos_pesos):
        print(str(indice) + ": " + arquivo)
    escolha_pesos = int(input('Escolha os pesos:'))
    arquivo_pesos = arquivos_pesos[escolha_pesos]
    caminho_pesos = '%s%s' % (pasta_pesos, arquivo_pesos)

    # define quais itens de teste usar
    print("-- Sempre será somente o primeiro acorde do compasso --\n"
          "1. Primeira nota do compasso\n"
          "2. Primeira e segunda nota do compasso\n"
          "3. Primeira metade das notas do compasso\n"
          "4. Todas as notas e pausas\n"
          "5. Todas as notas (sem pausas)")
    processamento = input('Defina o conjunto de dados para usar: ')
        
    caminho_processados = 'dados/blocos_teste'+processamento+'/*.npy'
    caminho_geracoes = "geracoes/"

    return caminho_modelo, caminho_pesos, caminho_processados, caminho_geracoes
    

def harmonizador():
    dicio_acordes = ['C:maj',  'C:min',  'C:7',  
                     'C#:maj', 'C#:min', 'C#:7', 
                     'D:maj',  'D:min',  'D:7',  
                     'D#:maj', 'D#:min', 'D#:7', 
                     'E:maj',  'E:min',  'E:7',  
                     'F:maj',  'F:min',  'F:7',  
                     'F#:maj', 'F#:min', 'F#:7', 
                     'G:maj',  'G:min',  'G:7',  
                     'G#:maj', 'G#:min', 'G#:7', 
                     'A:maj',  'A:min',  'A:7',  
                     'A#:maj', 'A#:min', 'A#:7', 
                     'B:maj',  'B:min',  'B:7']

    # carrega o modelo
    caminho_modelo, caminho_pesos, caminho_processados, caminho_geracoes = carregamento()
    modelo = model_from_json(open(caminho_modelo).read())
    modelo.load_weights(caminho_pesos)

    # compila novamente o modelo (mesma forma que no treino)
    modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # futura pasta para salvar geracoes
    if not os.path.isdir(caminho_geracoes):
        os.mkdir(caminho_geracoes)

    processados = glob.glob(caminho_processados)
    # iteracao de predicao para cada musica do arquivo de testes
    for musica in processados:
        sequencia_notas = sequence.pad_sequences(
            np.load(musica, allow_pickle=True), maxlen=modelo.input_shape[1])  # maxlen=32

        # realiza a predicao
        lista_predicao = []
        saida_rede = modelo.predict(sequencia_notas)
        # entender
        for acorde_indice in saida_rede.argmax(axis=1):
            lista_predicao.append(dicio_acordes[acorde_indice])

        # print comum das predicoes
        print(ntpath.basename(musica), lista_predicao)

        # salva as predicoes em arquivos CSV separados
        with open(caminho_geracoes+ntpath.basename(musica).split('.')[0]+'.csv', 'w') as arquivoCSV:
            escritor = csv.writer(arquivoCSV)
            for linha in lista_predicao: escritor.writerows([[linha]])

        arquivoCSV.close()

if __name__ == '__main__':
    harmonizador()
