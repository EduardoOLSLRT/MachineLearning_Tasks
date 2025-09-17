#====================================================
#------------------- Questão 1 -------------------
#====================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Dados de exemplo: Ritmo (X1), Energia (X2), Gênero (y)
# Gênero: 0 para Rock, 1 para Pop
data = {
    'Ritmo': [120, 125, 110, 95, 130, 90, 100, 115, 80, 140, 105, 118, 92, 135, 85],
    'Energia': [0.8, 0.9, 0.7, 0.4, 0.95, 0.3, 0.5, 0.75, 0.2, 0.88, 0.6, 0.82, 0.35, 0.93, 0.25],
    'Genero': [0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

X = df[['Ritmo', 'Energia']] # Features
y = df['Genero']           # Target

# 1. Dividir os dados em conjuntos de treino e teste (20% para teste)
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Inicializar o modelo classificador (Ex: Regressão Logística)
modelo = LogisticRegression()

# 3. Treinar o modelo
modelo.fit(X_treino, y_treino)

# 4. Fazer previsões no conjunto de teste
previsoes = modelo.predict(X_teste)

# 5. Calcular e imprimir a acurácia do modelo
acuracia = accuracy_score(y_teste, previsoes)
print(f"Acurácia do modelo: {acuracia:.2f}")

#====================================================
#------------------- Questão 2 -------------------
#====================================================

from sklearn.linear_model import LinearRegression

# Criando os arrays com numpy
# X: features (área e número de quartos)
X = np.array([
    [60, 2],   # Apto 1
    [75, 3],   # Apto 2
    [90, 3],   # Apto 3
    [120, 4]   # Apto 4
])

# y: preços em milhares de reais
y = np.array([200, 280, 350, 450])

# Inicializando o modelo de regressão linear
modelo_regressao = LinearRegression()

# Treinando o modelo com os dados fornecidos
modelo_regressao.fit(X, y)

# Fazendo previsão para um apartamento com 85 m² e 3 quartos
novo_apartamento = np.array([[85, 3]])
preco_previsto = modelo_regressao.predict(novo_apartamento)

# Imprimindo o preço previsto
print(f"Preço previsto para apartamento de 85 m² e 3 quartos: R$ {preco_previsto[0]:.0f} mil")

#====================================================
#------------------- EXPLICAÇÃO -------------------
#====================================================

# EXPLICAÇÃO DAS PERGUNTAS:
# ================================================

# 1. Por que este é um problema de aprendizado supervisionado?
# -----------------------------------------------------------
# Este é um problema de aprendizado porque temos dados históricos
# com respostas conhecidas (os preços reais dos apartamentos) que servem como
# "supervisão" para treinar o modelo. O algoritmo aprende a relação entre as
# características dos apartamentos, área e número de quartos, e seus preços;
# permitindo fazer previsões para novos apartamentos.

# 2. O que as variáveis X e Y representam no seu código?
# -----------------------------------------------------------
#   X : Representa as características de entrada - neste caso, a área
#   e o número de quartos de cada apartamento.
#   Y : Representa a resposta - os preços dos apartamentos

# 3. O que a função .fit() fez e o que a função .predict() fez?
# -----------------------------------------------------------
#   fit(): Treinou o modelo, analisando os dados de entrada (X) e as respostas
#   corretas (y) para encontrar a melhor linha reta que relaciona o apartamento e preço
#   predict(): Utilizou o modelo treinado para fazer uma previsão do preço de
#   um novo apartamento com características que o modelo nunca viu antes
