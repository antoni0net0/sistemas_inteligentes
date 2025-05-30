import pandas as pd

colunas = [
    "i",     # id
    "si1",   # pressão sistólica
    "si2",   # pressão diastólica
    "si3",   # qualidade da pressão
    "si4",   # pulso
    "si5",   # respiração
    "gi",    # gravidade
    "yi"     # classe (rótulo)
]

df = pd.read_csv('dataset/treino_sinais_vitais_com_label.txt', header=None, names=colunas)

dataset = df.drop(columns=["si1", "si2", "yi", "gi"])
rotulo = df["yi"]
gravidade = df["gi"]

quantidade_por_classe = df["yi"].value_counts()
#print(quantidade_por_classe)
