# PROJETO FINAL DE CURSO - RITA VITÓRIA
    # TREINAMENTO + AVALIAÇÃO + CONVERSÃO & EXPORTAÇÃO

# BIBLIOTECAS
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Desativa avisos do Tensor Flow

import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt 

from sklearn.tree import _tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 

# CONFIGURAÇÃO DO SEED - Garante a reprodutibilidade dos resultados
np.random.seed(42)
random.seed(42) 
tf.random.set_seed(42) 

# SALVAMENTO DE RESULTADOS
PASTA_BASE = r"C:\Users\ritav\SensorVirtual"
PASTA_RESULTADOS = os.path.join(PASTA_BASE, "resultados")
PASTA_GRAFICOS = os.path.join(PASTA_BASE, "graficos")

os.makedirs(PASTA_RESULTADOS, exist_ok=True)
os.makedirs(PASTA_GRAFICOS, exist_ok=True)

plt.rcParams.update({'font.size': 14})

# FUNÇÕES  
# ANÁLISE AMOSTRAS

def tabela_correlacao_massflow(): 
    df = pd.read_csv('ensaio1_dados_reais.csv').dropna() 
    
    correlacao = df.corr(numeric_only=True) 
    corr_massflow = correlacao[['Massflow(kg/s)']].sort_values( by='Massflow(kg/s)', ascending=False ) 
    corr_massflow.to_csv( os.path.join( PASTA_RESULTADOS, 'tabelacorrelacao_massflow.csv' ) ) 
    
    return corr_massflow

def analise_temporal():

    df = pd.read_csv('ensaio1_dados_reais.csv').dropna()

    tempo = np.arange(len(df))
    variaveis = ['ValveApt(%)', 'Temperature(K)', 'IntakeP(Pa)', 'OutputP(Pa)','Massflow(kg/s)']

    fig, axs = plt.subplots(len(variaveis), 2, figsize=(12, 10),dpi=300, sharex='col')

    for i, var in enumerate(variaveis):

        # Perfil temporal
        axs[i, 0].plot(tempo, df[var], color='darkgreen', linewidth=1)
        axs[i, 0].set_ylabel(var)
        axs[i, 0].grid(True)

        # Derivada
        derivada = np.gradient(df[var])

        axs[i, 1].plot(tempo, derivada, color='darkgreen', linewidth=1)
        axs[i, 1].grid(True)

    axs[0, 0].set_title('(a)')
    axs[0, 1].set_title('(b)')

    axs[-1, 0].set_xlabel('Amostras')
    axs[-1, 1].set_xlabel('Amostras')

    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, 'analise_temporal.png'), bbox_inches='tight')
    plt.close()

# DADOS
def tratamento_dados():
    df = pd.read_csv('ensaio1_dados_reais.csv').dropna()  # Carrega, lê e remove valores nulos
    df['tempo'] = np.arange(len(df)) * 0.05
    
    X = df[['ValveApt(%)', 'Temperature(K)', 'IntakeP(Pa)', 'OutputP(Pa)']].values # Variáveis de entrada
    Y = df['Massflow(kg/s)'].values # Variável alvo - saída que se quer prever
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # Separar sem "contaminação"

    scaler = MinMaxScaler() # Cria o "objeto normalizador"
    X_scaled_train = scaler.fit_transform(X_train) # Aprende a normalização e aplica nos dados de entrada de treino
    X_scaled_test = scaler.transform(X_test)  # Normaliza (coloca entre 0 e 1) os dados de entrada de teste

    # print("Máximos e mínimos das entradas normalizados:")
    # print("MIN:", [f"{v:.8f}" for v in scaler.data_min_])
    # print("MAX:", [f"{v:.8f}" for v in scaler.data_max_])
    
    return X_scaled_train, Y_train, X_scaled_test, Y_test

# MODELO RNA - MULTILAYER PERCEPTRON (MLP)
def treinamento_MLP(X_scaled_train, Y_train):
    inicio = time.time() # Tempo início do treinamento
    
    model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(4,)), 
            tf.keras.layers.Dense(8, activation='relu'), 
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary() # Mostra um resumo da arquitetura do modelo da rede (camadas, parâmetros etc.).

    history = model.fit(X_scaled_train, Y_train, epochs=400, batch_size=16, 
                        validation_split=0.2, verbose=2, shuffle=False)
    
    TEMPO = time.time() - inicio # Tempo de treinamento
    return model, history, TEMPO

# MODELO RANDOM FOREST
def treinamento_RF(X_scaled_train, Y_train):
    inicio = time.time() 
    
    rf = RandomForestRegressor(n_estimators=10, max_depth=5,
                               min_samples_leaf=4,random_state=42)

    rf.fit(X_scaled_train, Y_train) # Treina o modelo com os dados de treino
    
    TEMPO = time.time() - inicio # Tempo de treinamento
    return rf, TEMPO

# ANÁLISE DE DESEMPENHO (MÉTRICAS)
def avaliacao(nome, Y_train, Y_pred_train, Y_test, Y_pred_test, tempo, tamanho=None):
    return {
            "Modelo": nome, "R2_treino": r2_score(Y_train, Y_pred_train), 
            "R2_teste": r2_score(Y_test, Y_pred_test),
            "MSE": mean_squared_error(Y_test, Y_pred_test), 
            "MAE": mean_absolute_error(Y_test, Y_pred_test),   
            "Tempo Treinamento (s)": tempo, "Tamanho (KB)": tamanho
    }

# CONVERSÃO & EXPORTAÇÃO
def conversao_exportacao_MLP(model):
    caminho_base = r"C:\Users\ritav\SensorVirtual\embarcado_mlp"
    os.makedirs(caminho_base, exist_ok=True)

    caminho_tflite = os.path.join(caminho_base, "modelo_mlp.tflite")
    caminho_h = os.path.join(caminho_base, "mlp_model.h")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model) 
    tflite_model = converter.convert() # Converte o modelo para .tflite

    with open(caminho_tflite, "wb") as f:  # Cria o arquivo modelo_vazao.tflite no modo de escrita binária (wb) porque o modelo convertido é binário.
        f.write(tflite_model) # Escreve os bytes do modelo convertido no arquivo OU SEJA está pronto para ser embarcado em dispositivos com poucos recursos.

    tamanho_kb = len(tflite_model) / 1024 
    print(f"[OK] Arquivo modelo_mlp.tflite gerado com sucesso! Tamanho: {tamanho_kb:.8f} KB")

    # Converter para .h (Arduino)
    with open(caminho_tflite, "rb") as f:
        data = f.read()

    with open(caminho_h, "w") as f:
        f.write("const unsigned char model[] = {")
        f.write(",".join(str(b) for b in data))
        f.write("};")
    
    print(f"[OK] Arquivo model.h gerado com sucesso!")
    return tamanho_kb

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

    caminho_h = os.path.join(caminho_base, "rf_model.h")

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
    print(f"[OK] Arquivo rf_model.h gerado com sucesso! Tamanho: {tamanho_kb:.8f} KB")

    return tamanho_kb

# GRÁFICOS
def grafico_realpreditivo(Y_test, Y_pred_MLP, Y_pred_RF):
    plt.figure(figsize=(7,5), dpi=300)

    plt.plot([Y_test.min(), Y_test.max()],
             [Y_test.min(), Y_test.max()],
             'k--', label='Ideal') # Linha de ajuste ideal

    plt.scatter(Y_test, Y_pred_MLP, alpha=0.5, color='blue', label='MLP')
    plt.scatter(Y_test, Y_pred_RF, alpha=0.5, color='red', label='Random Forest')

    #plt.title('Comparativo dos modelos: Real vs Predito')
    plt.xlabel('Valor Real')
    plt.ylabel('Valor Predito')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, "grafico_comparativo_realpreditivo.png"))
    plt.close()

def grafico_erropredicao(Y_test, Y_pred_MLP, Y_pred_RF):
    erro_MLP =  Y_test - Y_pred_MLP.flatten() # Calcula diferença entre valor real e previsto
    erro_RF =  Y_test - Y_pred_RF.flatten() 

    plt.figure(figsize=(7,5), dpi=300)
    
    plt.hist(erro_MLP, bins=30, alpha=0.6, color='blue', label='Erro MLP')
    plt.hist(erro_RF, bins=30, alpha=0.6, color='red', label='Erro Random Forest')
    
    plt.axvline(0, color='k', linestyle='--') # Linha de referência para erro zero
    
    #plt.title(f'Distribuição do Erro de Predição')
    plt.xlabel('Erro')
    plt.ylabel('Frequência')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, f"grafico_comparativo_erropredicao.png"))
    plt.close()

def loss(history, nome_modelo):
    history_dict = history.history # Dicionário com os valores de perda e métricas durante o treinamento
    history_df = pd.DataFrame(history.history) # Dicionário para DataFrame
    history_df['epoch'] = history_df.index + 1 # Adiciona uma coluna de épocas
    history_df = history_df[['epoch'] + [col for col in history_df.columns if col != 'epoch']] # Coloca epoch no início

    # Curva de Aprendizado por Otimização (Treinamento)
    plt.figure(figsize=(7,5), dpi=300)
    plt.plot(history_dict['loss'], label='Treinamento', color='blue')
    plt.plot(history_dict['val_loss'], label='Validação de Treinamento', color='magenta', linestyle='--')
    
    #plt.title(f'Curva de Aprendizado - {nome_modelo}') 
    plt.xlabel('Epoca (Iteração)')
    plt.ylabel('Perda (Loss)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, f"grafico_curva_aprendizado_{nome_modelo}.png"))
    plt.close()

def aprendizado_RF(X_scaled_train, Y_train, X_scaled_test, Y_test, nome_modelo):
    
    max_train = X_scaled_train.shape[0]
    tamanhos = np.linspace(50, max_train, 10, dtype=int)
    r2_scores = []

    for n in tamanhos:

        X_sub = X_scaled_train[:n]
        Y_sub = Y_train[:n]

        rf = RandomForestRegressor(n_estimators=10, max_depth=5,
            min_samples_leaf=4, random_state=42
        )

        rf.fit(X_sub, Y_sub)
        Y_pred_test = rf.predict(X_scaled_test)

        r2_scores.append(r2_score(Y_test, Y_pred_test))

    # Gráfico
    plt.figure(figsize=(7,5), dpi=300)
    plt.plot(tamanhos, r2_scores, marker='o', color='red')

    #plt.title(f"Curva de Aprendizado - {nome_modelo}")
    plt.xlabel("Número de amostras de treino")
    plt.ylabel("R² teste")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, f"grafico_curva_aprendizado_{nome_modelo}.png"))
    plt.close()

# PIPELINE PRINCIPAL
resultados = [] 

tabela_correlacao_massflow()
analise_temporal()

X_scaled_train, Y_train, X_scaled_test, Y_test = tratamento_dados()

# Multilayer Perceptron (MLP)
model_MLP, history_MLP, TEMPO_MLP = treinamento_MLP(X_scaled_train, Y_train)
Y_pred_train_MLP = model_MLP.predict(X_scaled_train) # Saída Preditiva Treinamento
Y_pred_test_MLP = model_MLP.predict(X_scaled_test) # Saída Preditiva Validação (Dados de Teste)
tamanho_MLP = conversao_exportacao_MLP(model_MLP)

resultados.append(
    avaliacao(nome="MLP", Y_train=Y_train, Y_pred_train=Y_pred_train_MLP, 
                Y_test=Y_test, Y_pred_test=Y_pred_test_MLP,
                tempo=TEMPO_MLP, tamanho=tamanho_MLP
    )
)

# Random Forest
rf_model, TEMPO_RF = treinamento_RF(X_scaled_train, Y_train)
Y_pred_train_RF = rf_model.predict(X_scaled_train) 
Y_pred_test_RF = rf_model.predict(X_scaled_test)
tamanho_RF = conversao_exportacao_RF(rf_model)

resultados.append(
    avaliacao(nome="Random Forest", Y_train=Y_train, Y_pred_train=Y_pred_train_RF,
            Y_test=Y_test, Y_pred_test=Y_pred_test_RF,
            tempo=TEMPO_RF, tamanho=tamanho_RF,
    )
)

# Gráficos 
grafico_realpreditivo(Y_test, Y_pred_test_MLP, Y_pred_test_RF)
grafico_erropredicao(Y_test, Y_pred_test_MLP, Y_pred_test_RF)
loss(history_MLP, "MLP")
aprendizado_RF(X_scaled_train, Y_train, X_scaled_test, Y_test, "Random Forest")

# Arquivo .csv com métricas dos modelos
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv(f"{PASTA_RESULTADOS}/metricas_modelos.csv", index=False)

print("\nMétricas salvas!")
print("[OK] Execução finalizada com sucesso!")