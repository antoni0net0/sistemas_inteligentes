# Repositório de Modelos de Classificação

Este repositório contém implementações de três modelos de classificação (Árvore de Decisão ID3, Multi-Layer Perceptron e Random Forest) aplicados a um conjunto de dados de sinais vitais.

## Estrutura do Projeto

O projeto está organizado da seguinte forma:
# Repositório de Modelos de Classificação

Este repositório contém implementações de três modelos de classificação (Árvore de Decisão ID3, Multi-Layer Perceptron e Random Forest) aplicados a um conjunto de dados de sinais vitais.

## Estrutura do Projeto

O projeto está organizado da seguinte forma:
.
├── dataset.py
├── id3.py
├── mlp.py
├── random_forest.py
└── README.md

- `dataset.py`: Contém o código para carregar e pré-processar o conjunto de dados.
- `id3.py`: Implementa o modelo de Árvore de Decisão ID3 com validação cruzada.
- `mlp.py`: Implementa o modelo Multi-Layer Perceptron (Rede Neural Artificial) com validação cruzada.
- `random_forest.py`: Implementa o modelo Random Forest com validação cruzada.
- `README.md`: Este arquivo.

## Modelos Implementados

### 1. Árvore de Decisão (ID3)

A implementação da Árvore de Decisão utiliza o critério de entropia para a divisão dos nós. O modelo é avaliado usando validação cruzada estratificada de 5 folds e também em um conjunto de teste separado.

**Bibliotecas Utilizadas:**
- `pandas`
- `sklearn`

**Métricas de Avaliação:**
- Acurácia
- Matriz de Confusão
- Relatório de Classificação (Precision, Recall, F1-Score)

### 2. Multi-Layer Perceptron (MLP)

O MLP é construído usando a biblioteca TensorFlow/Keras. O modelo inclui camadas densas com ativação ReLU e uma camada de saída com ativação softmax. É aplicada normalização dos dados e validação cruzada K-Fold para avaliação.

**Arquitetura do Modelo:**
- Camada Densa (64 neurônios, ativação 'relu')
- Dropout (0.3)
- Camada Densa (32 neurônios, ativação 'relu')
- Camada Densa (número de classes, ativação 'softmax')

**Parâmetros de Treinamento:**
- Otimizador: Adam (learning rate=0.001)
- Função de Perda: `categorical_crossentropy`
- Épocas: 100
- Batch Size: 8

**Bibliotecas Utilizadas:**
- `numpy`
- `tensorflow`
- `sklearn`

**Métricas de Avaliação:**
- Acurácia Média nos Folds
- Desvio Padrão das Acurácias
- Matriz de Confusão
- Relatório de Classificação

### 3. Random Forest

O modelo Random Forest é um classificador de ensemble que utiliza múltiplas árvores de decisão. A implementação usa 100 estimadores e é avaliada com validação cruzada K-Fold. A normalização dos dados é realizada antes do treinamento.

**Parâmetros do Modelo:**
- `n_estimators`: 100
- `random_state`: 42

**Bibliotecas Utilizadas:**
- `numpy`
- `sklearn`

**Métricas de Avaliação:**
- Acurácia Média nos Folds
- Desvio Padrão das Acurácias
- Matriz de Confusão
- Relatório de Classificação

## Como Executar

Para executar os modelos, siga os passos abaixo:

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
    cd seu-repositorio
    ```

2.  **Crie um ambiente virtual (opcional, mas recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Linux/macOS
    venv\Scripts\activate     # No Windows
    ```

3.  **Instale as dependências:**
    Certifique-se de ter as bibliotecas necessárias instaladas. Você pode instalá-las via pip:
    ```bash
    pip install pandas scikit-learn tensorflow numpy
    ```

4.  **Prepare o dataset:**
    O código espera um arquivo `treino_sinais_vitais_com_label.txt` dentro de uma pasta `dataset` na raiz do projeto. Crie a pasta e coloque o arquivo lá:
    ```
    seu-repositorio/
    └── dataset/
        └── treino_sinais_vitais_com_label.txt
    ```
    As colunas esperadas no dataset são: `i`, `si1`, `si2`, `si3`, `si4`, `si5`, `gi`, `yi`. As colunas `si1`, `si2`, `yi`, `gi` são removidas do conjunto de dados de features para o treinamento, e `yi` é usada como rótulo.

5.  **Execute os scripts dos modelos:**

    - **Árvore de Decisão (ID3):**
      ```bash
      python id3.py
      ```

    - **Multi-Layer Perceptron (MLP):**
      ```bash
      python mlp.py
      ```

    - **Random Forest:**
      ```bash
      python random_forest.py
      ```

Cada script imprimirá no console os resultados da avaliação do respectivo modelo.
