# HARMONIZAÇÃO MUSICAL AUTOMÁTICA BASEADA EM REDES NEURAIS ARTIFICIAIS
 
Código fonte utilizado durante o desenvolvimento do Trabalho de Conclusão de Curso (TCC) de Lucas Costa, conforme intitulado.

Todo o sistema opera com base no banco de dados [CSV Leadsheet Database](http://marg.snu.ac.kr/chord_generation/), que deve ser disponibilizado na pasta `dados`.

A sequência de execução deve ser:

- `processamento.py`: simplifica e padroniza as informações do banco de dados;
- `construir_blocos`: define os conjutos de seleção de dados, explicados na Subseção 5.2.2 do TCC;
- `treino.py`: treina os modelos de Redes Neurais para os conjuntos de dados desejados;
- `harmonizador.py`: realiza a harmonização do conjunto de músicas de teste.

Nas pastas `json_modelo` e `pesos_modelo`, usadas para armazenar informações de treino dos modelos, estão os modelos originais utilizados para obter os resultados apresentados no TCC, podendo ser usados para reprodutibilidade de comportamento.
