import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#=======================================================
#================== EXERCICIO 1 ========================
#=======================================================
print("\n--- 1.2: Exercício - Prever Pontuação ---")
horas_jogadas = np.array([1, 3, 5, 8, 10]).reshape(-1, 1)
# Aqui a função "reshape" vai transformar cada número em uma nova lista
pontuacao_final = np.array([10, 25, 60, 90, 110])

modelo_pontuacao = LinearRegression()
modelo_pontuacao.fit(horas_jogadas, pontuacao_final)
horas_novas = np.array([[7]])
pontuacao_prevista = modelo_pontuacao.predict(horas_novas)
print(f"Para 7 horas jogadas, a pontuação prevista é de {pontuacao_prevista[0]:.0f} mil pontos.")

#=======================================================
#================== EXERCICIO 2 ========================
#=======================================================
print("\n--- 1.3: Exercício - Prever Temperatura ---")
altitudes = np.array([500, 1000, 1500, 2000, 3000]).reshape(-1, 1)
temperaturas = np.array([25, 20, 15, 10, 5])

modelo_temp = LinearRegression()
modelo_temp.fit(altitudes, temperaturas)
altitude_nova = np.array([[2500]])
temp_prevista = modelo_temp.predict(altitude_nova)
print(f"A temperatura prevista a 2500 metros é de {temp_prevista[0]:.1f}°C.")

#=======================================================
#================== EXERCICIO 3 ========================
#=======================================================
print("\n--- 2.2: Exercício - Aprovado/Reprovado ---")
notas_alunos = np.array([[8, 7], [5, 4], [9, 8], [4, 2], [7, 9], [3, 5]])
situacao = np.array([1, 0, 1, 0, 1, 0])

modelo_alunos = KNeighborsClassifier(n_neighbors=3)
# Aqui eu digo para ele usar os 3 mais próximos para classificar
modelo_alunos.fit(notas_alunos, situacao)
aluno_novo = np.array([[6, 7]])
previsao_aluno = modelo_alunos.predict(aluno_novo)
resultado_aluno = "Aprovado" if previsao_aluno[0] == 1 else "Reprovado"
print(f"Um aluno com notas [6, 7] foi classificado como: {resultado_aluno}")

#=======================================================
#================== EXERCICIO 4 ========================
#=======================================================
print("\n--- 2.3: Exercício - Classificar Veículo ---")
dados_veiculos = np.array([[150, 2], [1500, 4], [8000, 6], [180, 2], [2000, 4], [10000, 8]])
tipo_veiculo = np.array([0, 1, 2, 0, 1, 2])

modelo_veiculo = KNeighborsClassifier(n_neighbors=3)
modelo_veiculo.fit(dados_veiculos, tipo_veiculo)
veiculo_novo = np.array([[1800, 4]])
previsao_veiculo = modelo_veiculo.predict(veiculo_novo)
mapa_veiculos = {0: 'Moto', 1: 'Carro', 2: 'Caminhão'}
resultado_veiculo = mapa_veiculos[previsao_veiculo[0]]
print(f"Um veículo de [1800 kg, 4 rodas] foi classificado como: {resultado_veiculo}")

#=======================================================
#================== EXERCICIO 5 ========================
#=======================================================
print("\n--- 3.2: Exercício - Aprovar Empréstimo ---")
dados_credito = np.array([[50, 1], [30, 0], [80, 1], [40, 0], [120, 1], [70, 0]])
decisao_credito = np.array([1, 0, 1, 0, 1, 1])

modelo_credito = DecisionTreeClassifier(random_state=42)
# Aqui o decision avalia os dados e cria uma regra "se" e "random_state"
# é usado para criar números aleatorios e garantir que o resultado seja 
# o mesmo em cada execução
modelo_credito.fit(dados_credito, decisao_credito)
novo_cliente = np.array([[90, 1]])
previsao_credito = modelo_credito.predict(novo_cliente)
resultado_credito = "Aprovado" if previsao_credito[0] == 1 else "Negado"
print(f"Decisão para o cliente [R$90k, Casa Própria]: {resultado_credito}")

#=======================================================
#================== EXERCICIO 6 ========================
#=======================================================
print("\n--- 3.3: Exercício - Diagnóstico Médico ---")
sintomas = np.array([[1, 1], [0, 0], [1, 0], [0, 1], [1, 1], [0, 0]])
diagnostico = np.array([1, 0, 0, 0, 1, 0])

modelo_saude = DecisionTreeClassifier(random_state=42)
modelo_saude.fit(sintomas, diagnostico)
novo_paciente = np.array([[1, 0]])
previsao_saude = modelo_saude.predict(novo_paciente)
mapa_diagnostico = {0: 'Resfriado', 1: 'Gripe'}
resultado_saude = mapa_diagnostico[previsao_saude[0]]
print(f"Diagnóstico para [Febre=Sim, Tosse=Leve]: {resultado_saude}")