//Exercíco 1://

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# 1. Criar dataset simulado
np.random.seed(42)  # para reprodutibilidade

n_samples = 100

# Features numéricas
idade = np.random.randint(25, 66, n_samples)
salario_anual = np.random.randint(30000, 200001, n_samples).astype(float)
anos_empregado = np.random.randint(0, 31, n_samples)
valor_emprestimo = np.random.randint(5000, 100001, n_samples)

# Introduzindo valores ausentes na coluna salario_anual (~10%)
mask_nan = np.random.rand(n_samples) < 0.1
salario_anual[mask_nan] = np.nan

# Features categóricas
tipos_moradia = ['Aluguel', 'Propria', 'Financiada']
tipo_moradia = np.random.choice(tipos_moradia, n_samples)

# Alvo (target): risco_inadimplencia (0 = Baixo Risco, 1 = Alto Risco)
# Simulei um padrão simples: clientes mais jovens, com menor salário e empréstimos maiores têm mais risco.
risco_inadimplencia = (
    (idade < 40).astype(int) +
    (salario_anual < 70000).astype(int) +
    (valor_emprestimo > 50000).astype(int)
)
# Risco se soma >= 2
risco_inadimplencia = (risco_inadimplencia >= 2).astype(int)

# Criar DataFrame
df = pd.DataFrame({
    'idade': idade,
    'salario_anual': salario_anual,
    'anos_empregado': anos_empregado,
    'valor_emprestimo': valor_emprestimo,
    'tipo_moradia': tipo_moradia,
    'risco_inadimplencia': risco_inadimplencia
})

# 2. Separar features e target
X = df.drop('risco_inadimplencia', axis=1)
y = df['risco_inadimplencia']

# 3. Dividir em treino e teste (30% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 4. Definir pré-processamento
numeric_features = ['idade', 'salario_anual', 'anos_empregado', 'valor_emprestimo']
categorical_features = ['tipo_moradia']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),   # preencher NaNs com média
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 5. Criar pipeline final com o modelo
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

# 6. Treinar o modelo
model.fit(X_train, y_train)

# 7. Fazer previsões
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

# 8. Avaliar o modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

//Exercício 2://

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Criar dataset simulado
np.random.seed(42)
n_dias = 365  # 1 ano de dados

# Criar coluna de datas
datas = pd.date_range(start='2024-01-01', periods=n_dias, freq='D')

# Temperatura média simulada (varia ao longo do ano)
temperatura_media = 10 + 15 * np.sin(2 * np.pi * datas.dayofyear / 365) + np.random.normal(0, 2, n_dias)
temperatura_media = np.clip(temperatura_media, 0, 35)  # manter entre 0 e 35 graus

# Dia útil: 1 se segunda a sexta, 0 se fim de semana
dia_util = datas.weekday < 5  # segunda=0, domingo=6
dia_util = dia_util.astype(int)

# Consumo de energia simulado (relacionado à temperatura e se é dia útil)
consumo_energia_kwh = (
    3000 +
    50 * temperatura_media +           # dias quentes consomem mais (ar-condicionado)
    400 * dia_util +                   # dias úteis consomem mais
    np.random.normal(0, 200, n_dias)   # ruído
)

# Criar DataFrame
df = pd.DataFrame({
    'data': datas,
    'temperatura_media': temperatura_media,
    'dia_util': dia_util,
    'consumo_energia_kwh': consumo_energia_kwh
})

# 2. Engenharia de atributos a partir da data
df['mes'] = df['data'].dt.month
df['dia_da_semana'] = df['data'].dt.weekday  # 0 = segunda
df['dia_do_ano'] = df['data'].dt.dayofyear

# Remover a coluna de data original
df = df.drop('data', axis=1)

# 3. Separar X e y
X = df.drop('consumo_energia_kwh', axis=1)
y = df['consumo_energia_kwh']

# Dividir em treino e teste (30% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Criar pipeline de regressão
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(random_state=42))
])

# 5. Treinar o modelo
pipeline.fit(X_train, y_train)

# 6. Fazer previsões
y_pred = pipeline.predict(X_test)

# 7. Avaliação do modelo
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Erro Absoluto Médio (MAE): {mae:.2f} kWh")
print(f"R² Score: {r2:.3f}")

//Exércicio 3://

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Criando o dataset
dados = pd.DataFrame({
    'texto': [
        "oferta imperdível clique aqui agora",
        "ganhe dinheiro fácil",
        "relatório de vendas anexo",
        "oi, tudo bem? reunião amanhã",
        "promoção limitada, compre já",
        "envie seu currículo para vaga",
        "parabéns, você ganhou um prêmio",
        "atualização do sistema disponível",
        "não perca essa chance única",
        "reunião de equipe adiada"
    ],
    'categoria': [
        'spam', 'spam', 'ham', 'ham',
        'spam', 'ham', 'spam', 'ham',
        'spam', 'ham'
    ]
})

# 2. Separando features e alvo
X = dados['texto']
y = dados['categoria']

# 3. Dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. Criando pipeline com TfidfVectorizer e MultinomialNB
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# 5. Treinando o modelo
pipeline.fit(X_train, y_train)

# 6. Fazendo previsões no conjunto de teste
y_pred = pipeline.predict(X_test)

# 7. Avaliando o modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# 8. Comentário sobre alta precisão vs alto recall em spam
print("""
💡 Considerações:
- Para filtro de spam, é geralmente mais importante ter um alto recall para a classe 'spam'.
- Isso porque queremos pegar a maior parte dos spams para não deixar passar lixo na caixa de entrada.
- Porém, devemos equilibrar para não classificar muitos e-mails legítimos (ham) como spam, o que causa transtorno.
""")


//DESAFIOS//

//Exércico 4//
# Projeto 4: Classificação de Espécies de Flores Íris (Versão Aprimorada)
# Autor: [Seu Nome]
# Objetivo: Classificar automaticamente a espécie de uma flor Íris com base em medidas de pétalas e sépalas.
#           Agora com ajuste de hiperparâmetro via GridSearchCV.

# ==========================
# 📦 1. Importação das Bibliotecas
# ==========================
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# 🌸 2. Carregamento do Dataset
# ==========================
iris = load_iris()
X = iris.data
y = iris.target
classes = iris.target_names

print("📋 Informações do Dataset Iris:")
print(f"Features: {iris.feature_names}")
print(f"Classes: {classes}")
print(f"Tamanho total: {X.shape[0]} amostras\n")

# ==========================
# 🔀 3. Divisão em Treino e Teste
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# 🧩 4. Pipeline (StandardScaler + KNN)
# ==========================
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# ==========================
# 🔧 5. Ajuste Automático de n_neighbors com GridSearchCV
# ==========================
param_grid = {
    'knn__n_neighbors': [1, 3, 5, 7, 9, 11, 13],
    'knn__weights': ['uniform', 'distance']
}

print("🔍 Buscando o melhor número de vizinhos (n_neighbors)...")
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

# Melhor modelo encontrado
melhor_modelo = grid.best_estimator_
print("\n✅ Melhor combinação de parâmetros encontrada:")
print(grid.best_params_)

# ==========================
# ⚙️ 6. Treinamento com os melhores parâmetros
# ==========================
melhor_modelo.fit(X_train, y_train)

# ==========================
# 🔍 7. Avaliação do Modelo
# ==========================
y_pred = melhor_modelo.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Acurácia do Modelo Otimizado: {accuracy:.3f}\n")

print("📊 Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=classes))

# ==========================
# 📈 8. Visualização: Matriz de Confusão
# ==========================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=classes, yticklabels=classes)
plt.title("🌼 Matriz de Confusão - Modelo KNN Otimizado")
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")
plt.show()

# ==========================
# 💬 9. Interpretação dos Resultados
# ==========================
print("💡 Interpretação:")
if accuracy >= 0.97:
    print("Excelente! O modelo classifica quase todas as flores corretamente.")
elif accuracy >= 0.9:
    print("Bom desempenho! Pequenos ajustes em n_neighbors ou dados extras podem melhorar ainda mais.")
else:
    print("O desempenho é razoável. Considere testar outros algoritmos, como SVM ou Random Forest.")

//Exercício 5://
