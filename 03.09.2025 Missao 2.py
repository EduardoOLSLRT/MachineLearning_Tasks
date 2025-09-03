

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

#=====================================================================
#=====================================================================
print("--- Exercício 1 -  Missão 2 (Aprendizado Supervisionado) ---")
#=====================================================================
#=====================================================================

# Dados: [nota_prova_1, nota_trabalho_2]
# Rótulos: 0 = Reprovou, 1 = Passou
# Cloque notas no seu DataSet
X_treino = np.array([
    [8, 7], [9, 8], [7, 9], [8, 8], # Passou
    [3, 4], [2, 3], [4, 2], [3, 3]  # Reprovou
])
y_treino = np.array([1, 1, 1, 1, 0, 0, 0, 0])

# Criando o modelo. O KNN decide o rótulo de um novo ponto olhando para seus vizinhos mais próximos.
# n_neighbors=3 significa que ele vai consultar os 3 vizinhos mais próximos.
modelo_knn = KNeighborsClassifier(n_neighbors=3)

# O '.fit()' é onúcleo do aprendizado: o modelo analisa os dados e ajusta seus parâmetros.
modelo_knn.fit(X_treino, y_treino)

# testar com novos alunos.
aluno_A = np.array([[7, 6]]) # Esperamos que passe (1)
aluno_B = np.array([[5, 5]]) # Esperamos que reprove (0)

# O '.predict()' usa o conhecimento adquirido para fazer uma previsão.
previsao_A = modelo_knn.predict(aluno_A)
previsao_B = modelo_knn.predict(aluno_B)

print(f"Dados de treino (Notas): \n{X_treino}")
print(f"Rótulos de treino (Situação): {y_treino}")
print("-" * 20)
print(f"Previsão para o Aluno A: {'Passou' if previsao_A[0] == 1 else 'Reprovou'}")
print(f"Previsão para o Aluno B: {'Passou' if previsao_B[0] == 1 else 'Reprovou'}")
print("-" * 50, "\n")


from sklearn.linear_model import LinearRegression

#=====================================================================
#=====================================================================
print("--- Exercício 2 -  Missão 2 (Aprendizado Supervisionado) ---")
#=====================================================================
#=====================================================================

# Dados: [área_m2, numero_quartos]
# Rótulos: preco_em_milhares_de_reais
X_imoveis = np.array([
    [60, 2], [75, 3], [80, 3], # Imóveis menores
    [120, 3], [150, 4], [200, 4] # Imóveis maiores
])
y_precos = np.array([100, 150, 200, 250, 280, 320])

# TODO: Crie uma instância do modelo LinearRegression.
modelo_regressao = LinearRegression()

# TODO: Treine o modelo com os dados de imóveis (X_imoveis, y_precos).
modelo_regressao.fit(X_imoveis, y_precos)

# TODO: Crie um novo imóvel para testar (ex: 100m², 3 quartos).
imovel_teste = np.array([[100, 3]])

# TODO: Faça a previsão do preço para o novo imóvel.
preco_previsto = modelo_regressao.predict(imovel_teste)

print(f"Previsão de preço para um imóvel de 100m² com 3 quartos: R$ {preco_previsto[0]:.2f} mil")
print("-" * 50, "\n")

#=====================================================================
#=====================================================================
print("--- Exercício 3 -  Missão 2 (Aprendizado Supervisionado) ---\n")
#=====================================================================
#=====================================================================

from sklearn.cluster import KMeans

# Dados: [valor_gasto_medio, frequencia_visitas_mensal]
# Não temos rótulos!
clientes = np.array([
    [30, 1], [45, 2], [35, 1], # Grupo de baixo valor/frequência
    [500, 8], [600, 10], [550, 9] # Grupo de alto valor/frequência
])

# Criamos o modelo KMeans e pedimos para ele encontrar 2 clusters. Parametrize da forma que seja melhor (ver outros exemplos)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)

# '.fit_predict()' treina o modelo e já retorna o cluster de cada cliente.
clusters_encontrados = kmeans.fit_predict(clientes)

print(f"Dados dos clientes (sem rótulos):\n{clientes}")
print(f"Clusters encontrados pelo KMeans para cada cliente: {clusters_encontrados}")
print("Observe como o algoritmo separou corretamente os clientes nos grupos 0 e 1.")
print("-" * 50, "\n")


#=====================================================================
#=====================================================================
print("--- Exercício 4 -  Missão 2 (Aprendizado Não Supervisionado) ---")
#=====================================================================
#=====================================================================

from sklearn.linear_model import LinearRegression

# Dados: [valor_transacao, hora_do_dia (0-23)]
transacoes = np.array([
    [15.50, 14], [30.00, 10], [12.75, 11],
    [50.20, 19], [25.00, 9],
    [2500.00, 3] # Uma transação muito alta e de madrugada -> suspeita
])

# TODO: Crie um modelo KMeans para encontrar 2 grupos.
# A ideia é que as transações normais fiquem em um grupo e a anômala fique sozinha no outro.
modelo_anomalia = KMeans(n_clusters=2, random_state=42, n_init=10)

# TODO: Treine e preveja os clusters para os dados de transações.
clusters_transacoes = modelo_anomalia.fit_predict(transacoes)

print(f"Clusters para as transações: {clusters_transacoes}")
print("A transação anômala é aquela que está em um cluster isolado!\n")

#==============================================================================
#==============================================================================
# EXERCICIO FINAL DA MISSÃO 2 - APRENDIZADO POR REFORÇO
# 
# Imagine que você está ensinando um Doguinho a buscar um petisco. Você não escreve um manual de instruções para ele. Em vez disso, o processo é interativo:
# 
# 1 - O cachorro (o Agente) está em um ambiente (a sala). Ele decide fazer uma Ação (andar para frente).
# 2 - Você dá um Feedback (a Recompensa). Se ele chegou mais perto do petisco, você diz "Bom garoto!" (recompensa positiva).
# 3 - Se ele foi para longe, você não diz nada (recompensa neutra ou negativa, como o custo de energia).
# 4 - O cachorro recebe esse feedback e atualiza o seu "entendimento" sobre qual ação é boa naquela situação.
# 5 - Ele repete esse ciclo de Ação -> Recompensa várias vezes. Depois de muitas tentativas e erros, o cachorro aprende a sequência de ações ideal para conseguir o petisco da forma mais rápida possível.
# 
# O Aprendizado por Reforço é exatamente isso: um método de Machine Learning onde um agente aprende a tomar decisões em um ambiente para maximizar uma recompensa total ao longo do tempo.
# 
# Exercício: Agente Comilão
# Cenário: Nosso agente está em uma linha com 5 posições (0, 1, 2, 3, 4).
# O Agente: Um programa que só pode se mover para a 'direita'.
# O Ambiente: O caminho de 5 posições.
# O Objetivo: Chegar na Comida, que está na posição 4.
# 
# Regras de Recompensa:
# +20 pontos: Se o agente chegar na Comida.
# -1 ponto: Para cada passo que o agente der (representa o custo de energia).
# 
# Sua Missão: Você vai preencher a lógica do ambiente. O agente sempre tentará se mover para a 'direita'. Você precisa atualizar a posição dele, verificar se ele alcançou a comida e calcular a recompensa a cada passo.
#==============================================================================
#==============================================================================

print("\n--- Exercício 5 - Aprendizado por Reforço (Agente Comilão) ---\n")

class AmbienteComida:
    def __init__(self):
        self.posicao_comida = 4  # Comida está na posição 4
        self.posicao_agente = 0  # Agente começa na posição 0
        self.passos = 0
        self.recompensa_total = 0
    
    def reset(self):
        """Reseta o ambiente para o estado inicial"""
        self.posicao_agente = 0
        self.passos = 0
        self.recompensa_total = 0
        return self.posicao_agente
    
    def step(self, acao):
        """Executa uma ação e retorna (nova_posicao, recompensa, terminou)"""
        # O agente só pode se mover para a direita (ação = 1)
        if acao == 1 and self.posicao_agente < 4:
            self.posicao_agente += 1
            self.passos += 1
        
        # Calcula a recompensa
        recompensa = -1  # Custo de energia por passo
        
        # Verifica se chegou na comida
        terminou = False
        if self.posicao_agente == self.posicao_comida:
            recompensa = 20  # Recompensa por chegar na comida
            terminou = True
        
        self.recompensa_total += recompensa
        
        return self.posicao_agente, recompensa, terminou
    
    def render(self):
        """Mostra o estado atual do ambiente"""
        linha = ['_'] * 5
        linha[self.posicao_agente] = '🐕'  # Agente
        linha[self.posicao_comida] = '🍖'   # Comida
        print(f"Ambiente: {' '.join(linha)} | Posição: {self.posicao_agente} | Passos: {self.passos}")

# Criando o ambiente
ambiente = AmbienteComida()

print("Simulando o Agente Comilão...")
print("🐕 = Agente, 🍖 = Comida, _ = Posição vazia\n")

# Simulação de uma tentativa
print("=== Tentativa 1 ===")
ambiente.reset()
ambiente.render()

for passo in range(10):  # Máximo de 10 passos
    posicao, recompensa, terminou = ambiente.step(1)  # Sempre move para direita
    ambiente.render()
    
    if terminou:
        print(f"🎉 Agente chegou na comida! Recompensa total: {ambiente.recompensa_total}")
        break
    elif passo == 9:
        print(f"❌ Agente não conseguiu chegar na comida em 10 passos")

print(f"\nRecompensa total: {ambiente.recompensa_total}")
print(f"Passos dados: {ambiente.passos}")
print("-" * 50)
