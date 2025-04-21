import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import os 

# Funções auxiliares
def matriz_de_distancias(n, pontos):
    distancias = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                distancias[i][j] = -1
            else:
                distancias[i][j] = math.sqrt(
                    (pontos[i][0] - pontos[j][0]) ** 2 +
                    (pontos[i][1] - pontos[j][1]) ** 2
                )
    return distancias

def funcao_objetivo_tsp(n, solucao, distancias):
    distancia_total = 0
    for i in range(n - 1):
        distancia_total += distancias[solucao[i]][solucao[i + 1]]
    distancia_total += distancias[solucao[-1]][solucao[0]]
    return distancia_total

def solucao_inicial_aleatoria(n):
    return np.random.permutation(n)

def trocar_dois_cidades(solucao, n):
    nova = solucao.copy()
    i, j = random.sample(range(1, n), 2)
    nova[i], nova[j] = nova[j], nova[i]
    return nova

def salvar_resultado_txt(resultados, caminho="resultados_tsp.txt"):
    with open(caminho, "w", encoding="utf-8") as f:
        for r in resultados:
            f.write(f"Algoritmo: {r['Algoritmo']}\n")
            f.write(f"Nº Pontos: {r['Nº Pontos']}\n")
            f.write(f"Melhor Custo: {r['Melhor Custo']:.2f}\n")
            f.write(f"Tempo Execução (s): {r['Tempo Execução (s)']:.2f}\n")
            f.write("-" * 30 + "\n")          

def plotar_convergencia(historico, titulo="Convergência da Solução"):
    plt.plot(historico, color='green')
    plt.title(titulo)
    plt.xlabel("Iterações")
    plt.ylabel("Melhor Custo Encontrado")
    plt.grid(True)
    plt.show()

def gerar_instancia_tsp(nome_arquivo, n):
    with open(nome_arquivo, "w") as arquivo:
        for _ in range(n):
            x = random.randint(-10 * n, 10 * n)
            y = random.randint(-10 * n, 10 * n)
            arquivo.write(f"{x} {y}\n")

def gerar_pontos_aleatorios(n, limite=200):
    pontos = []
    for _ in range(n):
        x = random.randint(-limite, limite)
        y = random.randint(-limite, limite)
        pontos.append((x, y))
    return np.array(pontos)

def salvar_grafico_convergencia(historico, algoritmo, n, pasta_base="graficos_tsp"):
    if not os.path.exists(pasta_base):
        os.makedirs(pasta_base)

    pasta_algoritmo = os.path.join(pasta_base, algoritmo.replace(" ", "_"))
    if not os.path.exists(pasta_algoritmo):
        os.makedirs(pasta_algoritmo)

    plt.figure()
    plt.plot(historico, color='blue' if algoritmo == "Têmpera Simualda" else 'orange')
    plt.title(f"Convergência ({algoritmo}) - {n} pontos")
    plt.xlabel("Iterações")
    plt.ylabel("Melhor Custo Encontrado")
    plt.grid(True)

    caminho_arquivo = os.path.join(pasta_algoritmo, f"{n}_pontos.png")
    plt.savefig(caminho_arquivo)
    plt.close()


def salvar_grafico_comparativo(historico_sa, historico_ag, n, pasta="comparacao_graficos_tsp"):
    if not os.path.exists(pasta):
        os.makedirs(pasta)

    plt.figure()
    plt.plot(historico_sa, label="Têmpera Simualda", color='blue')
    plt.plot(historico_ag, label="Busca Local AG", color='orange')
    plt.title(f"Comparação de Convergência - {n} pontos")
    plt.xlabel("Iterações")
    plt.ylabel("Melhor Custo Encontrado")
    plt.grid(True)
    plt.legend()
    caminho_arquivo = os.path.join(pasta, f"comparacao_{n}_pontos.png")
    plt.savefig(caminho_arquivo)
    plt.close()

# Algoritmos
def tempera_simulada_tsp(pontos, temp_inicial=1000, temp_final=1.5, alpha=0.95, max_iter=2000):
    n = len(pontos)
    distancias = matriz_de_distancias(n, pontos)

    solucao = solucao_inicial_aleatoria(n)
    custo = funcao_objetivo_tsp(n, solucao, distancias)

    melhor_solucao = solucao.copy()
    melhor_custo = custo

    historico_melhores = np.zeros(max_iter)
    temperatura = temp_inicial

    for iteracao in range(max_iter):
        nova_solucao = trocar_dois_cidades(solucao, n)
        novo_custo = funcao_objetivo_tsp(n, nova_solucao, distancias)

        delta = novo_custo - custo
        if delta < 0 or random.random() < math.exp(-delta / temperatura):
            solucao = nova_solucao
            custo = novo_custo

        if custo < melhor_custo:
            melhor_custo = custo
            melhor_solucao = solucao.copy()

        historico_melhores[iteracao] = melhor_custo
        temperatura = temperatura * alpha if temperatura > temp_final else temp_inicial

    return melhor_solucao, melhor_custo, historico_melhores


def crossover(pai1, pai2, n):
    ponto_corte = random.randint(1, n - 2)
    filho1 = np.zeros(n, dtype=int)
    filho2 = np.zeros(n, dtype=int)

    filho1[:ponto_corte] = pai1[:ponto_corte]
    filho2[:ponto_corte] = pai2[:ponto_corte]

    k = ponto_corte
    for i in range(ponto_corte, n):
        if pai2[i] not in filho1:
            filho1[k] = pai2[i]
            k += 1
    
    k = ponto_corte
    for i in range(ponto_corte, n):
        if pai1[i] not in filho2:
            filho2[k] = pai1[i]
            k += 1

    return filho1, filho2


def mutacao(solucao, n, taxa_mutacao=0.05):
    if random.random() < taxa_mutacao:
        i, j = random.sample(range(n), 2)
        solucao[i], solucao[j] = solucao[j], solucao[i]
    return solucao


def selecao(populacao, distancias, n, tamanho_torneio=3):
    participantes = random.sample(populacao, tamanho_torneio)
    melhor_individuo = participantes[0]
    melhor_custo = funcao_objetivo_tsp(n, melhor_individuo, distancias)
    
    for individuo in participantes[1:]:
        custo = funcao_objetivo_tsp(n, individuo, distancias)
        if custo < melhor_custo:
            melhor_individuo = individuo
            melhor_custo = custo
            
    return melhor_individuo

def selecao_torneio(populacao, custos, tamanho_torneio=3):
    participantes = random.sample(populacao, tamanho_torneio)
    melhor_individuo = participantes[0]
    melhor_custo = custos[populacao.index(melhor_individuo)]

    for individuo in participantes[1:]:
        custo = custos[populacao.index(individuo)]
        if custo < melhor_custo:
            melhor_individuo = individuo
            melhor_custo = custo

    return melhor_individuo


def algoritmo_genetico(pontos, max_iter=2000, tamanho_populacao=100, taxa_mutacao=0.05, tamanho_torneio=3):
    n = len(pontos)
    distancias = matriz_de_distancias(n, pontos)

    populacao = [solucao_inicial_aleatoria(n) for _ in range(tamanho_populacao)]

    melhor_solucao = None
    melhor_custo = float('inf')

    historico_melhores = np.zeros(max_iter)

    for iteracao in range(max_iter):
        nova_populacao = []

        melhor_da_geracao = min(populacao, key=lambda x: funcao_objetivo_tsp(n, x, distancias))
        custo_melhor = funcao_objetivo_tsp(n, melhor_da_geracao, distancias)
        
        if custo_melhor < melhor_custo:
            melhor_solucao = melhor_da_geracao
            melhor_custo = custo_melhor

        while len(nova_populacao) < tamanho_populacao:
            pai1 = selecao(populacao, distancias, n, tamanho_torneio)
            pai2 = selecao(populacao, distancias, n, tamanho_torneio)
            
            filho1, filho2 = crossover(pai1, pai2, n)
            filho1 = mutacao(filho1, n, taxa_mutacao)
            filho2 = mutacao(filho2, n, taxa_mutacao)
            
            nova_populacao.append(filho1)
            nova_populacao.append(filho2)
        
        populacao = nova_populacao

        historico_melhores[iteracao] = melhor_custo

    return melhor_solucao, melhor_custo, historico_melhores

def main():
    tamanhos = [50, 100, 200, 500, 1000, 2000, 5000]
    resultados = []

    for n in tamanhos:
        print(f"\n[INFO] Testando com {n} pontos...")

        pontos = gerar_pontos_aleatorios(n)
        
        # Têmpera Simulada
        inicio_sa = time.time()
        melhor_sa, custo_sa, historico_sa = tempera_simulada_tsp(pontos)
        tempo_sa = time.time() - inicio_sa
        print(f"SA - Custo: {custo_sa:.2f}, Tempo: {tempo_sa:.2f}s")

        resultados.append({
            "Algoritmo": "Têmpera Simulada",
            "Nº Pontos": n,
            "Melhor Custo": custo_sa,
            "Tempo Execução (s)": tempo_sa
        })
        salvar_grafico_convergencia(historico_sa, "Têmpera Simulada", n)

        # Algoritmo Genético
        inicio_ag = time.time()
        melhor_ag, custo_ag, historico_ag = algoritmo_genetico(pontos)
        tempo_ag = time.time() - inicio_ag
        print(f"AG - Custo: {custo_ag:.2f}, Tempo: {tempo_ag:.2f}s")

        resultados.append({
            "Algoritmo": "Algoritmo Genético",
            "Nº Pontos": n,
            "Melhor Custo": custo_ag,
            "Tempo Execução (s)": tempo_ag
        })
        salvar_grafico_convergencia(historico_ag, "Algoritmo Genético", n)
        
        salvar_grafico_comparativo(historico_sa, historico_ag, n)

    salvar_resultado_txt(resultados)
    tabela = pd.DataFrame(resultados)
    tabela.to_excel("comparacao_algoritmos_tsp.xlsx", index=False)

# Execução
if __name__ == "__main__":
    main()