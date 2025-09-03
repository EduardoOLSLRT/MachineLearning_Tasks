# TAREFAS DE APRENDIZADO SUPERVISIONADO E NÃO SUPERVISIONADO
# Implementação completa de todas as 6 tarefas

import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

print("=" * 60)
print("TAREFAS DE APRENDIZADO SUPERVISIONADO E NÃO SUPERVISIONADO")
print("=" * 60)

# =============================================================================
# TAREFA 1: Treine o modelo abaixo colocando mais 4 conjuntos de dados
# =============================================================================
print("\n" + "=" * 50)
print("TAREFA 1: Classificador de Intenções de Mensagens")
print("=" * 50)

# Função de pré-processamento
def limpar_texto(texto):
    texto = texto.lower()  # Converte para minúsculas
    texto = re.sub(r'[^\w\s]', '', texto)  # Remove pontuação
    texto = re.sub(r'\d+', '', texto)  # Remove números
    texto = texto.strip()  # Remove espaços extras
    return texto

# 1. Conjunto de dados expandido (mensagens + rótulos)
mensagens = [
    "Quero fazer um pedido",
    "Preciso falar com o suporte",
    "Quais promoções vocês têm hoje?",
    "Qual o horário de funcionamento?",
    "Meu produto veio com defeito",
    "Posso pagar com cartão de crédito?",
    # 4 novos conjuntos de dados adicionados
    "Gostaria de cancelar minha compra",
    "Como faço para rastrear meu pedido?",
    "Vocês têm desconto para estudantes?",
    "Preciso trocar o tamanho da roupa",
    "Qual a forma de pagamento aceita?",
    "Meu código promocional não funcionou",
    "Quero saber sobre a política de devolução",
    "Como faço para criar uma conta?",
    "Preciso de ajuda com o login",
    "Quais são os prazos de entrega?",
    "Vocês fazem entrega no meu bairro?"
]
rotulos = ["pedido", "suporte", "promoção", "informação", "suporte", "pagamento",
           "cancelamento", "rastreamento", "promoção", "troca", "pagamento",
           "promoção", "devolução", "conta", "suporte", "entrega", "entrega"]

# 2. Pré-processamento das mensagens
mensagens_limpas = [limpar_texto(m) for m in mensagens]

# 3. Vetorização
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(mensagens_limpas)

# 4. Treinamento do modelo
modelo = MultinomialNB()
modelo.fit(X, rotulos)

print("Modelo treinado com sucesso!")
print(f"Total de mensagens de treinamento: {len(mensagens)}")

# 5. Teste do modelo
print("\n--- Teste do Modelo ---")
testes = [
    "Quero comprar algo",
    "Preciso de ajuda urgente",
    "Tem cupom de desconto?",
    "Qual o horário da loja?",
    "Meu produto quebrou"
]

for teste in testes:
    teste_limpo = limpar_texto(teste)
    X_teste = vectorizer.transform([teste_limpo])
    predicao = modelo.predict(X_teste)
    print(f"'{teste}' -> Intenção: {predicao[0]}")

# 6. Interação com o usuário
print("\n--- Interação Interativa ---")
while True:
    nova_mensagem = input("\nDigite uma mensagem (ou 'sair' para encerrar): ")
    if nova_mensagem.lower() == "sair":
        break
    nova_mensagem_limpa = limpar_texto(nova_mensagem)
    X_novo = vectorizer.transform([nova_mensagem_limpa])
    predicao = modelo.predict(X_novo)
    print(f"Intenção prevista: {predicao[0]}")

# =============================================================================
# TAREFA 2: Classificador de mensagens para bot de atendimento acadêmico
# =============================================================================
print("\n" + "=" * 50)
print("TAREFA 2: Bot de Atendimento Acadêmico")
print("=" * 50)

# 1. Dataset acadêmico
frases_academicas = [
    "Quando abre matrícula?",
    "Preciso de segunda chamada",
    "Qual o horário da biblioteca?",
    "Como faço para pegar o histórico escolar?",
    "Quais são as datas dos exames?",
    "Preciso de ajuda com o sistema acadêmico",
    "Qual o prazo para entrega do TCC?",
    "Como faço para solicitar transferência?",
    "Quais são os eventos da semana acadêmica?",
    "Preciso de orientação sobre estágio",
    "Como faço para cancelar matrícula?",
    "Qual o horário de atendimento da secretaria?",
    "Preciso de declaração de matrícula",
    "Como faço para solicitar bolsa de estudos?",
    "Quais são as regras para colação de grau?"
]
rotulos_academicos = [
    "matricula", "avaliacao", "biblioteca", "documentos", "avaliacao",
    "sistema", "tcc", "transferencia", "eventos", "estagio",
    "matricula", "secretaria", "documentos", "bolsa", "formatura"
]

# 2. Vetorização
vectorizer_academico = CountVectorizer()
X_academico = vectorizer_academico.fit_transform(frases_academicas)

# 3. Modelo
modelo_academico = MultinomialNB()
modelo_academico.fit(X_academico, rotulos_academicos)

# 4. Teste
print("Modelo acadêmico treinado!")
teste_academico = "Preciso saber quando posso me matricular"
teste_limpo = limpar_texto(teste_academico)
X_teste_academico = vectorizer_academico.transform([teste_limpo])
predicao_academico = modelo_academico.predict(X_teste_academico)
print(f"Teste: '{teste_academico}' -> Categoria: {predicao_academico[0]}")

# =============================================================================
# TAREFA 3: Previsão de Tempo de Entrega de Pizza
# =============================================================================
print("\n" + "=" * 50)
print("TAREFA 3: Previsão de Tempo de Entrega de Pizza")
print("=" * 50)

# Dados de Treino: [distancia_km, numero_de_pizzas]
dados_entregas = np.array([
    [5, 2],   # 5 km, 2 pizzas
    [2, 1],   # 2 km, 1 pizza
    [10, 4],  # 10 km, 4 pizzas
    [7, 3],   # 7 km, 3 pizzas
    [1, 1],   # 1 km, 1 pizza
    [8, 2],   # 8 km, 2 pizzas (novo)
    [3, 1],   # 3 km, 1 pizza (novo)
    [12, 5],  # 12 km, 5 pizzas (novo)
    [4, 2],   # 4 km, 2 pizzas (novo)
    [15, 6]   # 15 km, 6 pizzas (novo)
])

# Rótulos: Tempo de entrega em minutos
tempos_entrega = np.array([30, 15, 55, 40, 10, 35, 20, 65, 25, 75])

# Criação e treinamento do modelo
modelo_entrega = LinearRegression()
modelo_entrega.fit(dados_entregas, tempos_entrega)

# Previsão para um novo pedido: 8 km de distância e 2 pizzas
pedido_novo = np.array([[8, 2]])
tempo_previsto = modelo_entrega.predict(pedido_novo)

print(f"Tempo de entrega previsto para o novo pedido: {tempo_previsto[0]:.2f} minutos")

# Teste com outros pedidos
testes_entrega = [
    [6, 3],   # 6 km, 3 pizzas
    [9, 1],   # 9 km, 1 pizza
    [11, 4]   # 11 km, 4 pizzas
]

print("\n--- Testes de Previsão de Entrega ---")
for i, teste in enumerate(testes_entrega):
    tempo = modelo_entrega.predict([teste])[0]
    print(f"Pedido {i+1}: {teste[0]} km, {teste[1]} pizza(s) -> {tempo:.2f} minutos")

# =============================================================================
# TAREFA 4: Agrupamento de Mensagens (K-Means com 3 clusters)
# =============================================================================
print("\n" + "=" * 50)
print("TAREFA 4: Agrupamento de Mensagens (3 Clusters)")
print("=" * 50)

# 1. Matriz de mensagens expandida (sem rótulos)
mensagens_cluster = [
    "Quero pedir pizza",
    "Qual o valor da pizza grande?",
    "Preciso de suporte no aplicativo",
    "O app está travando",
    "Vocês têm sobremesas?",
    "Meu pedido está atrasado",
    # 3 novos conjuntos de dados adicionados
    "Quero uma pizza margherita",
    "Qual o tamanho da pizza média?",
    "Preciso de ajuda para fazer login",
    "O site não está carregando",
    "Vocês têm bebidas?",
    "Meu pedido não chegou ainda",
    "Quero uma pizza pepperoni",
    "Qual o preço da pizza pequena?",
    "Como faço para cadastrar meu endereço?",
    "O aplicativo está com erro",
    "Vocês têm refrigerante?",
    "Meu pedido está demorando muito"
]

# 2. Vetorizar texto
vectorizer_cluster = CountVectorizer()
X_cluster = vectorizer_cluster.fit_transform(mensagens_cluster)

# 3. Criar modelo de agrupamento com 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_cluster)

# 4. Mostrar os grupos encontrados
print("\nAgrupamento de mensagens (3 clusters):")
for i, msg in enumerate(mensagens_cluster):
    print(f"'{msg}' => Cluster {kmeans.labels_[i]}")

# Análise dos clusters
print("\n--- Análise dos Clusters ---")
for cluster_id in range(3):
    cluster_msgs = [msg for i, msg in enumerate(mensagens_cluster) if kmeans.labels_[i] == cluster_id]
    print(f"\nCluster {cluster_id} ({len(cluster_msgs)} mensagens):")
    for msg in cluster_msgs:
        print(f"  - {msg}")

# 5. Interação: classificar nova frase
print("\n--- Interação: Classificar Nova Mensagem ---")
while True:
    nova_mensagem = input("\nDigite uma nova mensagem (ou 'sair' para encerrar): ")
    if nova_mensagem.lower() == "sair":
        break
    X_novo = vectorizer_cluster.transform([nova_mensagem])
    cluster_previsto = kmeans.predict(X_novo)
    print(f"Essa mensagem se parece com o Cluster {cluster_previsto[0]}")

# =============================================================================
# TAREFA 5: Agrupar frases de chatbot de turismo
# =============================================================================
print("\n" + "=" * 50)
print("TAREFA 5: Chatbot de Turismo")
print("=" * 50)

# 1. Dataset de turismo
frases_turismo = [
    "Quero reservar hotel",
    "Preciso de passagem aérea",
    "Quais são os melhores restaurantes?",
    "Quero fazer um passeio guiado",
    "Preciso de aluguel de carro",
    "Quais são os pontos turísticos?",
    "Quero reservar um hostel",
    "Preciso de passagem de ônibus",
    "Quais são os bares da região?",
    "Quero fazer um tour de barco",
    "Preciso de seguro viagem",
    "Quais são as praias próximas?",
    "Quero reservar um apartamento",
    "Preciso de passagem de trem",
    "Quais são os museus da cidade?",
    "Quero fazer um passeio de helicóptero",
    "Preciso de transfer do aeroporto",
    "Quais são os shoppings?",
    "Quero reservar uma pousada",
    "Preciso de passagem de barco"
]

# 2. Vetorização
vectorizer_turismo = CountVectorizer()
X_turismo = vectorizer_turismo.fit_transform(frases_turismo)

# 3. Modelo KMeans com 4 clusters
kmeans_turismo = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_turismo.fit(X_turismo)

# 4. Saída
print("Agrupamento de frases de turismo:")
for i, frase in enumerate(frases_turismo):
    print(f"'{frase}' => Cluster {kmeans_turismo.labels_[i]}")

# Análise dos clusters de turismo
print("\n--- Análise dos Clusters de Turismo ---")
categorias = ["Hospedagem", "Transporte", "Alimentação", "Passeios"]
for cluster_id in range(4):
    cluster_frases = [frase for i, frase in enumerate(frases_turismo) if kmeans_turismo.labels_[i] == cluster_id]
    print(f"\nCluster {cluster_id} - {categorias[cluster_id]} ({len(cluster_frases)} frases):")
    for frase in cluster_frases:
        print(f"  - {frase}")

# =============================================================================
# TAREFA 6: Encontrar Produtos "Âncora"
# =============================================================================
print("\n" + "=" * 50)
print("TAREFA 6: Produtos Âncora")
print("=" * 50)

# Dados: [preco_produto, nota_de_popularidade (0-10)]
dados_produtos = np.array([
    [10, 2], [15, 3], [12, 1],   # Categoria 1: Baratos e menos populares
    [200, 9], [180, 8], [210, 10], # Categoria 2: Caros e muito populares
    [50, 5], [45, 4], [55, 6],   # Categoria 3: Médios (novo)
    [300, 7], [280, 6], [320, 8] # Categoria 4: Premium (novo)
])

# Criação e treinamento do modelo KMeans para encontrar 2 clusters
modelo_produtos = KMeans(n_clusters=2, random_state=42, n_init=10)
modelo_produtos.fit(dados_produtos)

# Os centros dos clusters são os nossos produtos "âncora" ideais
produtos_ancora = modelo_produtos.cluster_centers_

print(f"Características dos Produtos Âncora (Preço, Popularidade):")
for i, produto in enumerate(produtos_ancora):
    print(f"Produto Âncora {i+1}: Preço R$ {produto[0]:.2f}, Popularidade {produto[1]:.2f}/10")

# Análise dos clusters
print("\n--- Análise dos Clusters de Produtos ---")
for cluster_id in range(2):
    cluster_produtos = [produto for i, produto in enumerate(dados_produtos) if modelo_produtos.labels_[i] == cluster_id]
    print(f"\nCluster {cluster_id} ({len(cluster_produtos)} produtos):")
    for produto in cluster_produtos:
        print(f"  - Preço: R$ {produto[0]}, Popularidade: {produto[1]}/10")

print("\n" + "=" * 60)
print("TODAS AS TAREFAS CONCLUÍDAS COM SUCESSO!")
print("=" * 60)
