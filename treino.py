from keras_preprocessing import sequence
from keras.layers import *
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import time
import os


def modelo_rnn_simples(tamanho_sequencia, dimensao_entrada, dimensao_saida, neuronios):
	# camada dimensionada de entrada (notas)
    entrada_camada = Input(
        shape=(tamanho_sequencia, dimensao_entrada), dtype='float32')
    # distribuicao dos termos no tempo
    # no caso de apenas uma nota, funciona apenas como Dense
    temporal_camada = TimeDistributed(Dense(dimensao_entrada))(entrada_camada)
    # primeiro dropout
    dropout1 = Dropout(0.2)(temporal_camada)
    # camada de rnn simples
    rnn_camada = SimpleRNN(neuronios, return_sequences=False)(dropout1)
    # segundo dropout
    dropout2 = Dropout(0.2)(rnn_camada)
    # uniao de resultados
    saida_rnn = Dense(neuronios, activation='relu')(dropout2)
    # terceiro dropout
    dropout3 = Dropout(0.2)(saida_rnn)

    # lambda e flatten nao fazem muito sentido, Dense dá a saída desejada
    #soma_camada = Lambda(lambda xin: K.sum(xin, axis=1))(rnn_camada)
    #soma_camada = Flatten()(dropout)

    # camada de saida com dimensao desejada (acordes)
    saida_camada = Dense(dimensao_saida, activation='softmax')(dropout3)

    # definicao do moledo
    modelo = Model(inputs=entrada_camada, outputs=saida_camada)
    # plot do formato da rede neural configurada
    plot_model(modelo, to_file='resultados/estrutura_rnn_{0}.png'.format(
        time.strftime("%Y%m%d_%H_%M")), show_shapes=True, expand_nested=True)
    return modelo


def modelo_lstm(tamanho_sequencia, dimensao_entrada, dimensao_saida, neuronios):
    # semelhante a rnn
    entrada_camada = Input(
        shape=(tamanho_sequencia, dimensao_entrada), dtype='float32')
    temporal_camada = TimeDistributed(Dense(dimensao_entrada))(entrada_camada)
    dropout1 = Dropout(0.5)(temporal_camada)

    # camada de lstm
    lstm_camada = LSTM(neuronios, return_sequences=False)(dropout1)
    dropout2 = Dropout(0.5)(lstm_camada)
    
    saida_lstm = Dense(neuronios, activation='relu')(dropout2)
    dropout3 = Dropout(0.5)(saida_lstm)
    
    saida_camada = Dense(dimensao_saida, activation='softmax')(dropout3)

    modelo = Model(inputs=entrada_camada, outputs=saida_camada)
    plot_model(modelo, to_file='resultados/estrutura_lstm_{0}.png'.format(
        time.strftime("%Y%m%d_%H_%M")), show_shapes=True, expand_nested=True)
    return modelo


def modelo_lstm_duplo(tamanho_sequencia, dimensao_entrada, dimensao_saida, neuronios):
    entrada_camada = Input(
        shape=(tamanho_sequencia, dimensao_entrada), dtype='float32')
    temporal_camada = TimeDistributed(Dense(dimensao_entrada))(entrada_camada)
    dropout1 = Dropout(0.2)(temporal_camada)
    # duas camadas conectadas de lstm, a primeira com return_sequences
    lstm_camada_1 = LSTM(neuronios, return_sequences=True)(dropout1)
    dropout2 = Dropout(0.2)(lstm_camada_1)
    # sem return_sequences, pois nao tem tempo
    lstm_camada_2 = LSTM(neuronios, return_sequences=False)(dropout2)
    dropout3 = Dropout(0.2)(lstm_camada_2)

    #soma_camada = Lambda(lambda xin: K.sum(xin, axis=1))(lstm_camada_2)

    saida_lstm = Dense(neuronios, activation='relu')(dropout3)
    dropout4 = Dropout(0.2)(saida_lstm)
    saida_camada = Dense(dimensao_saida, activation='softmax')(dropout4)

    modelo = Model(inputs=entrada_camada, outputs=saida_camada)
    plot_model(modelo, to_file='resultados/estrutura_lstm_dupla_{0}.png'.format(
        time.strftime("%Y%m%d_%H_%M")), show_shapes=True, expand_nested=True)
    return modelo


def resultados(history, entrada):
	# Resultados do treino
    print("Plotando historio de acuracia")
    # Plot valores de acuracia de treino e teste
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Acuracia do Modelo')
    plt.ylabel('Acuracia')
    plt.xlabel('Epochs')
    plt.legend(['Treino', 'Teste'], loc='upper left')
    plt.savefig('resultados/acuracia_modelo%s_%s.png' %
                (entrada, time.strftime("%Y%m%d_%H_%M")))
    print("Acuracia plotado")
    plt.close()

    print("Plotando historio de erro")
    # Plot valores de erro de treino e teste
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Erro do Modelo')
    plt.ylabel('Erro')
    plt.xlabel('Epochs')
    plt.legend(['Treino', 'Teste'], loc='upper left')
    plt.savefig('resultados/erro_modelo%s_%s.png' %
                (entrada, time.strftime("%Y%m%d_%H_%M")))
    print("Erro plotado")
    plt.close()


def treino():
    print("1. RNN Simples\n"
          "2. LSTM - uma camada\n"
          "3. LSTM - duas camadas")
    entrada = input('Escolha o modelo a ser treinado: ')

    print("-- Sempre será somente o primeiro acorde do compasso --\n"
          "1. Primeira nota do compasso\n"
          "2. Primeira e segunda nota do compasso\n"
          "3. Primeira metade das notas do compasso\n"
          "4. Todas as notas e pausas\n"
          "5. Todas as notas (sem pausas)")
    processamento = input('Selecione o modo de processamento: ')

    # Para garantir que todas as sequências tenham o mesmo comprimento.
    # Por padrão, isso é feito preenchendo 0 no início de cada sequência até que cada sequência tenha o
    # mesmo comprimento que a sequência mais longa.
    # Isso é necessário para os casos de dados que consideram mais de uma nota por compasso, por exemplo
    vetor_entrada_treino = sequence.pad_sequences(np.load(
        'dados/vetor_entrada_treino_processado{0}.npy'.format(processamento), allow_pickle=True))
    vetor_saida_treino = sequence.pad_sequences(np.load(
        'dados/vetor_saida_treino_processado{0}.npy'.format(processamento), allow_pickle=True))

    vetor_entrada_teste = sequence.pad_sequences(np.load(
        'dados/vetor_entrada_teste_processado{0}.npy'.format(processamento), allow_pickle=True), maxlen=vetor_entrada_treino.shape[1])
    vetor_saida_teste = sequence.pad_sequences(np.load(
        'dados/vetor_saida_teste_processado{0}.npy'.format(processamento), allow_pickle=True))

    # dimensoes de entrada para os modelos
    dimensao_entrada = vetor_entrada_treino.shape[2]
    dimensao_saida = vetor_saida_treino.shape[1]
    tamanho_sequencia = vetor_entrada_treino.shape[1]

    # Para avaliacao de hiperparametros
    porcentagem_entrada_treino = vetor_entrada_treino.shape[0] #// 10
    porcentagem_entrada_teste = vetor_entrada_teste.shape[0] #// 10

    # hiperparametros
    qtde_epochs = 3
    tamanho_lote = 512
    neuronios = 64

    # cria pasta de resultados
    pasta_resultados = 'resultados/'
    if not os.path.isdir(pasta_resultados):
        os.mkdir(pasta_resultados)

    # faz a modelagem de acordo com especificado
    if entrada == '1':
        modelo = modelo_rnn_simples(
            tamanho_sequencia, dimensao_entrada, dimensao_saida, neuronios)
    elif entrada == '2':
        modelo = modelo_lstm(
            tamanho_sequencia, dimensao_entrada, dimensao_saida, neuronios)
    elif entrada == '3':
        modelo = modelo_lstm_duplo(
            tamanho_sequencia, dimensao_entrada, dimensao_saida, neuronios)

    # print dos dados do modelo escolhido
    modelo.summary()
    # compilacao do modelo
    modelo.compile(loss='categorical_crossentropy',
                   optimizer='adam', metrics=['accuracy'])
    # variavel para acompanhar acuracia
    history = modelo.fit(vetor_entrada_treino[:porcentagem_entrada_treino], vetor_saida_treino[:porcentagem_entrada_treino],
                         batch_size=tamanho_lote, epochs=qtde_epochs, validation_data=(vetor_entrada_teste[:porcentagem_entrada_teste], vetor_saida_teste[:porcentagem_entrada_teste]))
    # acuracia total
    print("Calculando acuracia...")
    _, acuracia = modelo.evaluate(vetor_entrada_teste, vetor_saida_teste)
    print('Acuracia: %f' % (acuracia))

    # salva os pesos obtidos
    pasta_pesos = 'pesos_modelo/'
    if not os.path.isdir(pasta_pesos):
        os.mkdir(pasta_pesos)
    # cria arquivo com tipo do modelo e quantidade de epochs
    arquivo_pesos = 'modelo%s_processamento%s_%sepochs_%s' % (
        entrada, processamento, qtde_epochs, time.strftime("%Y%m%d_%H_%M.h5"))
    caminho_pesos = '%s%s' % (pasta_pesos, arquivo_pesos)
    modelo.save_weights(caminho_pesos)

    # salva o modelo
    modelo_json = modelo.to_json()
    pasta_modelo = 'json_modelo/'
    if not os.path.isdir(pasta_modelo):
        os.mkdir(pasta_modelo)
    # arquivo com tipo de modelo e epochs (bate com anterior)
    arquivo_modelo = 'modelo%s_processamento%s_%sepochs_%s' % (
        entrada, processamento, qtde_epochs, time.strftime("%Y%m%d_%H_%M.json"))
    caminho_modelo = '%s%s' % (pasta_modelo, arquivo_modelo)
    open(caminho_modelo, 'w').write(modelo_json)

    print("Fim do treino!")
    resultados(history, entrada)


if __name__ == '__main__':
    treino()
