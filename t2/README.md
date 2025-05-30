# Repositório de Modelos de Classificação

Este repositório contém implementações de três modelos de classificação (Árvore de Decisão ID3, Multi-Layer Perceptron e Random Forest) aplicados a um conjunto de dados de sinais vitais.

## Estrutura do Projeto

O projeto está organizado da seguinte forma:
# Repositório de Modelos de Classificação

Este repositório contém implementações de três modelos de classificação (Árvore de Decisão ID3, Multi-Layer Perceptron e Random Forest) aplicados a um conjunto de dados de sinais vitais.

## Estrutura do Projeto

O projeto está organizado da seguinte forma:
```bash
├── dataset.py
├── id3.py
├── mlp.py
├── random_forest.py
└── README.md
```
- `dataset.py`: Contém o código para carregar e pré-processar o conjunto de dados.
- `id3.py`: Implementa o modelo de Árvore de Decisão ID3.
- `mlp.py`: Implementa o modelo Multi-Layer Perceptron (Rede Neural Artificial).
- `random_forest.py`: Implementa o modelo Random Forest.
- `README.md`: Este arquivo.

## Modelos Implementados

### 1. Árvore de Decisão (ID3)

A implementação da Árvore de Decisão utiliza o critério de entropia para a divisão dos nós. O modelo é avaliado usando validação cruzada estratificada de 5 folds e também em um conjunto de teste separado.

### 2. Multi-Layer Perceptron (MLP)

O MLP é construído usando a biblioteca TensorFlow/Keras. O modelo inclui camadas densas com ativação ReLU e uma camada de saída com ativação softmax. É aplicada normalização dos dados e validação cruzada K-Fold para avaliação.

### 3. Random Forest

O modelo Random Forest é um classificador de ensemble que utiliza múltiplas árvores de decisão. A implementação usa 100 estimadores e é avaliada com validação cruzada K-Fold. A normalização dos dados é realizada antes do treinamento.

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
    **Importante:** Para garantir a compatibilidade com a biblioteca TensorFlow, é crucial utilizar uma versão do Python anterior à 3.10, pois versões mais recentes podem apresentar problemas de compatibilidade.
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
