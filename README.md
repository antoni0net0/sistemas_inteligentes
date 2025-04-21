# Comparação de Algoritmos para o Problema do Caixeiro Viajante (TSP)

Este projeto tem como objetivo comparar o desempenho de dois algoritmos para resolução do Problema do Caixeiro Viajante (TSP - Travelling Salesman Problem):

- **Têmpera Simulada (Simulated Annealing)**
- **Busca Local Aleatória com Trocas (AG)**

## Funcionalidades

- Geração de instâncias aleatórias com diferentes números de pontos.
- Execução dos algoritmos de otimização.
- Salvamento de gráficos de convergência individuais e comparativos.
- Exportação de resultados para arquivos `.txt` e `.xlsx`.

## Algoritmos Implementados

### Têmpera Simulada

Algoritmo inspirado no processo de resfriamento de metais, que aceita soluções piores com certa probabilidade para escapar de mínimos locais.

Parâmetros ajustáveis:
- Temperatura inicial e final
- Fator de resfriamento (alpha)
- Número máximo de iterações

### Busca Local AG

Algoritmo simples que tenta melhorar a solução trocando duas cidades de lugar, mantendo a troca somente se ela melhorar o custo total do percurso.

## Estrutura de Arquivos Gerados

- `resultados_tsp.txt`: resumo dos custos e tempos de execução dos algoritmos.
- `comparacao_algoritmos_tsp.xlsx`: tabela com os mesmos dados em formato planilha.
- Pastas com gráficos de convergência e comparações:
  - `graficos_tsp/Têmpera_Simualda/`
  - `graficos_tsp/Busca_Local_AG/`
  - `comparacao_graficos_tsp/`

## Dependências

- numpy
- matplotlib
- pandas

Você pode instalar as dependências com:

```bash
pip install numpy matplotlib pandas
```

## Execução

Para rodar o experimento completo:

```bash
python tsp_main.py
```
