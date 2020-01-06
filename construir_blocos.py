import glob
import csv
import os
import ntpath
import numpy as np


def codificacao_one_hot(length, one_index):
    # Retorna vetor codificado
    # vetor de zeros
    vetor = [0] * length
    # posicao do 1 no vetor
    vetor[one_index] = 1

    return vetor


def constroe_blocos_teste(nome_arquivo, lista_musicas, processamento, entrada):
    # Cria bloco para cada arquivo de musica do conjunto de teste
    caminho = "dados/blocos_teste"+str(processamento)
        
    if not os.path.isdir(caminho):
        os.mkdir(caminho)
    np.save('%s/%s.npy' %
            (caminho, nome_arquivo.split('.')[0]), np.array(lista_musicas))

def construtor():
    #threshold : int, optional
    #   Total number of array elements which trigger summarization rather than full repr (default 1000).
    np.set_printoptions(threshold=np.inf)

    # dicionario de notas e acordes: correspondem com todas possibilidades do banco de dados processado
    dicio_notas = ['C', 'C#', 'D', 'D#', 'E', 'F',
                   'F#', 'G', 'G#', 'A', 'A#', 'B', 'rest']
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

    print("1. Conjunto de treino\n"
          "2. Conjunto de teste")
    entrada = input('Selecione de qual conjunto criar bloco de dados: ')
    if entrada == '1':
        caminho = 'dados/novo_treino/*.csv'
    elif entrada == '2':
        caminho = 'dados/novo_teste/*.csv'
    else:
        print("Erro de entrada")
        return None

    print("-- Sempre será somente o primeiro acorde do compasso --\n"
          "1. Primeira nota do compasso\n"
          "2. Primeira e segunda nota do compasso\n"
          "3. Primeira metade das notas do compasso\n"
          "4. Todas as notas e pausas\n"
          "5. Todas as notas (sem pausas)")
    processamento = input('Selecione o modo de processamento: ')

    lista_opcoes = ['2', '3', '4', '5']

    arquivos_csv = glob.glob(caminho)
    # pega tamanho dos dicionarios
    dicio_notas_tamanho = len(dicio_notas)
    dicio_acordes_tamanho = len(dicio_acordes)

    # declaracao das listas usadas durante o processamento
    # a mesma lista de treino eh usada para processar o conjunto de teste
    matriz_resultado_treino = []
    matriz_resultado_validacao = []

    # construcao das matrizes a partir dos arquivos csv
    # logica como do processamento
    for indice, caminho_csv in enumerate(arquivos_csv):
        csv_aberto = open(caminho_csv, 'r', encoding='utf-8')
        next(csv_aberto)
        leitor = csv.reader(csv_aberto)

        # lista da sequencia de notas sendo processada
        sequencia_notas = []
        # lista de cada musica (bloco) processada
        lista_musicas = []
        # variavel de controle de compassos
        compasso_anterior = None
        duracao_soma = 0

        for linha in leitor:
            # processamento reduziu o banco de dados a 4 colunas de infomacao:
            #    0       1      2       3
            # measure, chord, note, duration
            compasso = int(linha[0])
            acorde = linha[1]
            nota = linha[2]
            duracao = float(linha[3])

            duracao_soma += duracao

            # pega posicao da nota no dicionario
            nota_indice = dicio_notas.index(nota)
            acorde_indice = dicio_acordes.index(acorde)

            # tecnica de codificacao "one-hot" (https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)
            cod_vetor_nota = codificacao_one_hot(
                dicio_notas_tamanho, nota_indice)
            cod_vetor_acorde = codificacao_one_hot(
                dicio_acordes_tamanho, acorde_indice)

            # ------------------------------------------------
            # ---------------- PROCESSAMENTO -----------------
            # ------------------------------------------------
            """ 
            MODO DE PROCESSAMENTO 1:
            - Salva o acorde do comeco de cada compasso
            - Salva a nota do comeco de cada compasso
            """
            if processamento == '1':
                # caso de ser a primeira linha do arquivo de musica
                if compasso_anterior is None or compasso_anterior != compasso:
                    # salva para as listas de musica
                    lista_musicas.append([cod_vetor_nota])
                    # salva a nota
                    matriz_resultado_treino.append([cod_vetor_nota])
                    # salva o acorde
                    matriz_resultado_validacao.append(cod_vetor_acorde)

                """ 
                MODO DE PROCESSAMENTO 2:
                - Salva o acorde do comeco de cada compasso
                - Salva a nota do comeco e do meio de cada compasso
                """
            elif processamento == '2':
                # caso de ser a primeira linha do arquivo de musica
                if compasso_anterior is None:
                    sequencia_notas.append(cod_vetor_nota)
                    # salva o acorde
                    matriz_resultado_validacao.append(cod_vetor_acorde)

                # caso de estar no meio do compasso
                elif compasso_anterior == compasso:
                    if duracao_soma == 8:
                        sequencia_notas.append(cod_vetor_nota)

                # quando muda de compasso
                else:
                    # sequencia de notas de cada musica
                    lista_musicas.append(sequencia_notas)
                    # salva notas do compasso anterior
                    matriz_resultado_treino.append(sequencia_notas)
                    # pega primeira nota do compasso corrente
                    sequencia_notas = [cod_vetor_nota]
                    # salva novo acorde
                    matriz_resultado_validacao.append(cod_vetor_acorde)
                    # reinicia a soma de duracoes
                    duracao_soma = 0

                """ 
                MODO DE PROCESSAMENTO 3:
                - Salva o acorde do comeco de cada compasso
                - Salva as notas do primeiro meio do compasso
                """
            elif processamento == '3':
                # caso de ser a primeira linha do arquivo de musica
                if compasso_anterior is None:
                    sequencia_notas.append(cod_vetor_nota)
                    # salva o acorde
                    matriz_resultado_validacao.append(cod_vetor_acorde)

                # caso de estar no meio do compasso
                elif compasso_anterior == compasso:
                    if duracao_soma <= 8:
                        sequencia_notas.append(cod_vetor_nota)

                # quando muda de compasso
                else:
                    # sequencia de notas de cada musica
                    lista_musicas.append(sequencia_notas)
                    # salva notas do compasso anterior
                    matriz_resultado_treino.append(sequencia_notas)
                    # pega primeira nota do compasso corrente
                    sequencia_notas = [cod_vetor_nota]
                    # salva novo acorde
                    matriz_resultado_validacao.append(cod_vetor_acorde)
                    # reinicia a soma de duracoes
                    duracao_soma = 0

                """ 
                MODO DE PROCESSAMENTO 4:
                - Salva o acorde do comeco de cada compasso
                - Salva todas as notas de cada compasso
                - Ignora silencios de compasso inteiro
                """
            elif processamento == '4':
                # caso de ser a primeira linha do arquivo de musica
                if compasso_anterior is None:
                    if nota == 'rest':
                        if duracao != 16.0:
                            sequencia_notas.append(cod_vetor_nota)
                    else:
                        sequencia_notas.append(cod_vetor_nota)

                    # salva o acorde
                    matriz_resultado_validacao.append(cod_vetor_acorde)

                # caso de estar ainda no mesmo compasso
                elif compasso_anterior == compasso:
                    if nota == 'rest':
                        if duracao != 16.0:
                            sequencia_notas.append(cod_vetor_nota)
                    else:
                        sequencia_notas.append(cod_vetor_nota)

                # quando muda de compasso
                else:
                    # sequencia de notas de cada musica
                    lista_musicas.append(sequencia_notas)
                    # salva notas do compasso anterior
                    matriz_resultado_treino.append(sequencia_notas)
                    # pega primeira nota do compasso corrente
                    if nota == 'rest':
                        if duracao != 16.0:
                            sequencia_notas = [cod_vetor_nota]
                        else:
                            sequencia_notas = []
                    else:
                        sequencia_notas = [cod_vetor_nota]

                    # salva novo acorde
                    matriz_resultado_validacao.append(cod_vetor_acorde)

                """ 
                MODO DE PROCESSAMENTO 5:
                - Salva o acorde do comeco de cada compasso
                - Salva todas as notas de cada compasso
                - Ignora silencios no geral
                """
            elif processamento == '5':
                # caso de ser a primeira linha do arquivo de musica
                if compasso_anterior is None:
                    if nota != 'rest':
                        sequencia_notas.append(cod_vetor_nota)

                    # salva o acorde
                    matriz_resultado_validacao.append(cod_vetor_acorde)

                # caso de estar ainda no mesmo compasso
                elif compasso_anterior == compasso:
                    if nota != 'rest':
                        sequencia_notas.append(cod_vetor_nota)

                # quando muda de compasso
                else:
                    # sequencia de notas de cada musica
                    lista_musicas.append(sequencia_notas)
                    # salva notas do compasso anterior
                    matriz_resultado_treino.append(sequencia_notas)
                    # pega primeira nota do compasso corrente
                    if nota != 'rest':
                        sequencia_notas = [cod_vetor_nota]
                    else:
                        sequencia_notas = []

                    # salva novo acorde
                    matriz_resultado_validacao.append(cod_vetor_acorde)

            # ------------------------------------------------
            # -------------FIM DO PROCESSAMENTO --------------
            # ------------------------------------------------

            # atualiza variavel de compasso e repete
            compasso_anterior = compasso

        # ultima sequencia de notas do ultimo compasso da musica atual
        if any(opcao == processamento for opcao in lista_opcoes):
            matriz_resultado_treino.append(sequencia_notas)
            lista_musicas.append(sequencia_notas)

        # no caso de processar dados de teste, salva individualmente as musicas (blocos) processadas
        if entrada == '2':
            constroe_blocos_teste(ntpath.basename(
                caminho_csv), lista_musicas, processamento, entrada)

    # salva arquivos de treino
    if entrada == '1':
        np.save('dados/vetor_entrada_treino_processado{0}.npy'.format(
            processamento), np.array(matriz_resultado_treino))
        np.save('dados/vetor_saida_treino_processado{0}.npy'.format(
            processamento), np.array(matriz_resultado_validacao))
    # salva compilado de teste
    elif entrada == '2':
        np.save('dados/vetor_entrada_teste_processado{0}.npy'.format(
            processamento), np.array(matriz_resultado_treino))
        np.save('dados/vetor_saida_teste_processado{0}.npy'.format(
            processamento), np.array(matriz_resultado_validacao))


if __name__ == '__main__':
    construtor()
