# PROJETO FINAL DE CURSO - RITA VITÓRIA
# ORIENTADOR: TÚLIO CHAVES

# BIBLIOTECAS
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Desativa mensagens de aviso do TensorFlow

import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt 

from sklearn.tree import _tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 

# SALVAMENTO DE RESULTADOS
PASTA_BASE = r"C:\Users\ritav\SensorVirtual"
PASTA_RESULTADOS = os.path.join(PASTA_BASE, "resultados")
PASTA_GRAFICOS = os.path.join(PASTA_BASE, "graficos")

os.makedirs(PASTA_RESULTADOS, exist_ok=True)
os.makedirs(PASTA_GRAFICOS, exist_ok=True)

# FUNÇÕES  
# DADOS
def tratamento_dados():
    df = pd.read_csv('ensaio1_dados_reais.csv').dropna()  # Carrega, lê e remove valores nulos
    X = df[['ValveApt(%)', 'Temperature(K)', 'IntakeP(Pa)', 'OutputP(Pa)']].values # Variáveis de entrada
    Y = df['Massflow(kg/s)'].values # Variável alvo - saída que se quer prever

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # Separar sem "contaminação"

    scaler = MinMaxScaler() # Cria o "objeto normalizador"
    X_scaled_train = scaler.fit_transform(X_train) # Aprende a normalização e aplica nos dados de entrada de treino
    X_scaled_test = scaler.transform(X_test)  # Normaliza (coloca entre 0 e 1) os dados de entrada de teste

    print("Máximos e mínimos das entradas normalizados:")
    print("MIN:", [f"{v:.8f}" for v in scaler.data_min_])
    print("MAX:", [f"{v:.8f}" for v in scaler.data_max_])

    return X_scaled_train, Y_train, X_scaled_test, Y_test

# MODELO RANDOM FOREST
def treinamento_RF(X_scaled_train, Y_train):
    inicio = time.time() 
    
    rf = RandomForestRegressor(random_state=42)
    # Hiperparâmetros (Busca)
    param_dist = {
        'n_estimators': [10, 50, 100, 200, 300], # Número de árvores
        'max_depth': [None, 5, 10, 20, 30, 40], # Profundidade máxima das árvores
        'min_samples_split': [2, 5, 10, 20], # Número mínimo de amostras para dividir um nó
        'min_samples_leaf': [1, 2, 4], # Número mínimo de amostras em um nó folha
        'max_features': ['sqrt', 'log2', 0.5, 0.7, 1.0] # Número de recursos a considerar para melhor divisão
    }

    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=30,
                                    cv=5, verbose=2, random_state=42, n_jobs=-1)

    random_search.fit(X_scaled_train, Y_train) # Busca pelos melhores hiperparâmetros e treina o modelo com eles
    print('Melhores parâmetros encontrados:', random_search.best_params_)
    best_model = random_search.best_estimator_ # Modelo com os melhores hiperparâmetros encontrados
    
    TEMPO = time.time() - inicio # Tempo de treinamento
    return best_model, TEMPO

# ANÁLISE DE DESEMPENHO (MÉTRICAS)
def avaliacao(nome, Y_train, Y_pred_train, Y_test, Y_pred_test, tempo, tamanho=None):
    return {
            "Modelo": nome, "R2_treino": r2_score(Y_train, Y_pred_train), 
            "R2_teste": r2_score(Y_test, Y_pred_test),
            "MSE": mean_squared_error(Y_test, Y_pred_test), 
            "MAE": mean_absolute_error(Y_test, Y_pred_test),   
            "Tempo Treinamento (s)": tempo, "Tamanho (KB)": tamanho
    }

# RF em if/else (uma árvore)
def conversao_c(tree, feature_names, nome_funcao):
    tree_ = tree.tree_

    def recurse(node):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]

            return f"""
        
if ({name} <= {threshold:.5f}) {{{recurse(tree_.children_left[node])}}} 
else {{{recurse(tree_.children_right[node])}}}"""
        else:
            value = tree_.value[node][0][0]
            return f"return {value:.5f};"

    return f"""
float {nome_funcao}(float ValveApt, float Temperature, float IntakeP, float OutputP) {{{recurse(0)}}}"""

# Modelo RF para .h (Arduino)
def conversao_exportacao_RF(rf_model):
    caminho_base = r"C:\Users\ritav\SensorVirtual\embarcado_rf"
    os.makedirs(caminho_base, exist_ok=True)

    caminho_h = os.path.join(caminho_base, "rf_bestmodel.h")

    feature_names = ['ValveApt', 'Temperature', 'IntakeP', 'OutputP']

    codigo = "#ifndef RF_MODEL_H\n#define RF_MODEL_H\n\n"

    # Gerar funções das árvores
    for i, tree in enumerate(rf_model.estimators_):
        codigo += conversao_c(tree, feature_names, f"arvore_{i}")

    codigo += "\nfloat predict(float v, float t, float ip, float op) {\n"
    codigo += "    float soma = 0;\n"

    for i in range(len(rf_model.estimators_)):
        codigo += f"    soma += arvore_{i}(v, t, ip, op);\n"

    codigo += f"    return soma / {len(rf_model.estimators_)}.0;\n"
    codigo += "}\n\n#endif"

    # Salvar arquivo
    with open(caminho_h, "w") as f:
        f.write(codigo)

    tamanho_kb = os.path.getsize(caminho_h) / 1024
    print(f"[OK] Arquivo rf_bestmodel.h gerado com sucesso! Tamanho: {tamanho_kb:.8f} KB")

    return tamanho_kb


# PIPELINE PRINCIPAL
resultados = [] 
X_scaled_train, Y_train, X_scaled_test, Y_test = tratamento_dados()

# Random Forest
rf_bestmodel, TEMPO_RF = treinamento_RF(X_scaled_train, Y_train)
Y_pred_train_RF = rf_bestmodel.predict(X_scaled_train) 
Y_pred_test_RF = rf_bestmodel.predict(X_scaled_test)
tamanho_RF = conversao_exportacao_RF(rf_bestmodel)

resultados.append(
    avaliacao(nome="Random Forest", Y_train=Y_train, Y_pred_train=Y_pred_train_RF,
            Y_test=Y_test, Y_pred_test=Y_pred_test_RF,
            tempo=TEMPO_RF, tamanho=tamanho_RF,
    )
)

# Arquivo .csv com métricas dos modelos
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv(f"{PASTA_RESULTADOS}/metricas_rfbest.csv", index=False)

print("\nMétricas salvas!")
print("[OK] Execução finalizada com sucesso!")