# REPRODUÇÃO DO TRABALHO DE JOÃO GABRIEL DRUMMOND
# SCRIPT PYTHON UTILIZADO PARA APRENDIZADO DE MÁQUINA (ML) - FLUXO COMPLETO

# Bibliotecas
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler  # Normaliza (coloca entre 0 e 1) os dados de entrada.
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # Funçõe que avaliam o desempenho do modelo (erros e precisão)
import matplotlib.pyplot as plt # Biblioteca para geração de gráficos
import os # Biblioteca para manipulação de arquivos e diretórios

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Desativa mensagens de aviso do TensorFlow (opcional, para deixar a saída mais limpa)

# Etapa 1 do ML - Carregamento e pré-processamento dos dados ----------------------------------------------------
print("-> Carregando dados...")
df = pd.read_csv('dataset_vazao.csv') # Leitura de dados

print("-> Limpando dados...")
df = df.dropna() # Remove linhas com valores nulos

print("-> Preparando variáveis de entrada e saída...")
X = df[['temperatura', 'pressao', 'posicao_valvula']].values # Variáveis preditoras - entradas usadas para previsão
y = df['vazao_massica'].values # Variável alvo - saída que se quer prever

# Normalizar as entradas
print("-> Normalizando entradas...")
scaler = MinMaxScaler() # Cria o "objeto normalizador"
X_scaled = scaler.fit_transform(X) # Aprende os valores min e max da variável X e aplica a fórmula de normalização

# Construção do modelo de RNA
print("-> Criando modelo...")
    # 1. Keras: é uma API (interface de alto nível) dentro do TensorFlow que serve para ...
    # ...criar e treinar redes neurais de forma mais simples e intuitiva.
    # 2. Layers (camadas): são as partes que compõem a rede neural onde cada camada realiza uma transformação...
    # ... nos dados. Dense = “totalmente conectada”: cada neurônio de uma camada está ligado a todos da anterior.
    # 3. Rede MLP (Multilayer Perceptron): é uma rede neural totalmente conectada com múltiplas camadas sendo...
    # ...o tipo mais básico e versátil usada para resolver problemas de regressão e classificação e etc...
    # ...Sua estrutura geral é: Camada de entrada -> Camadas oculta 1 -> Camadas oculta 2 -> Camada de saída.
    # 4. Funções de ativação: 
model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(3,)), # Define quantas entradas o modelo terá
    # Primeira camada oculta com 8 neurônios onde cada neurônio aplica uma operação matemática e passa o resultado... 
    # ...para a próxima camada. A função de ativação 'relu' ajuda o modelo a aprender relações não lineares.
tf.keras.layers.Dense(8, activation='relu'), 
    # Segunda camada oculta com 16 neurônio ajuda o modelo a capturar padrões não lineares mais complexos.
    # Função de ativação 'relu' novamente para aprender essas relações não lineares.
tf.keras.layers.Dense(16, activation='relu'),
    # Camada de saída com 1 neurônio que fornece a previsão final da vazão mássica.
    # Não usa ativação aqui porque é um problema de regressão (previsão de valor contínuo).
tf.keras.layers.Dense(1)])
    # Define como o modelo vai aprender com um método de ajuste de pesos (rápido e eficiente), função de perda...
    # ...(mede a diferença entre previsão e valor real) e métricas para avaliar o desempenho durante o treinamento..
    # ...(erro médio absoluto - MAE que mostra a média dos erros em módulo
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary() # Mostra um resumo da arquitetura do modelo da rede (camadas, parâmetros etc.).
print(" -> Treinando modelo...")
    # Realiza o treino do modelo repetindo o processo 400 vezes (épocas) sobre o conjunto de dados...
    #... processando 16 amostras por vez (batch size) e reservando 80% dos dados para treino e 20% para validação...
    #...(verificar se o modelo está aprendendo bem em dados que ele ainda não viu durante o treino evitando 
    #...que ele “decore" os dados (o chamado overfitting). Verbose=0 desativa a exibição do progresso do treinamento.
history = model.fit(X_scaled, y, epochs=400, batch_size=16, validation_split=0.2, verbose=2)

# ---------------------------------------------------------------------------------------------
print("-> Avaliando modelo...")
y_pred = model.predict(X_scaled) # Usa o modelo treinado para gerar previsões da variável alvo a partir das entradas normalizadas
print("MSE:", mean_squared_error(y, y_pred)) # Mede quanto o modelo erra em média, elevando os erros ao quadrado (Quanto menor o MSE, melhor o modelo)
print("MAE:", mean_absolute_error(y, y_pred)) # Mede a diferença média absoluta entre o valor real e o previsto. (Quanto menor o MAE, melhor o modelo)
print("R2 :", r2_score(y, y_pred)) # Coeficiente de determinação que indica a precisão do modelo (Quanto mais próximo de 1, melhor o modelo)

# ADICIONAL DA RITA -----------------------------------------------------------------------------
# Tabela de Métricas
MSE = mean_squared_error(y, y_pred)
MAE = mean_absolute_error(y, y_pred)
R2 = r2_score(y, y_pred)

metricas = [
    ["MSE", f"{MSE:.8f}", "0"],
    ["MAE", f"{MAE:.6f}", "0"],
    ["R²", f"{R2:.4f}", "1"]
]

colunas = ["Métrica", "Valor Obtido", "Valor Ideal"]

fig, ax = plt.subplots()
ax.axis('off')

tabela = ax.table(
    cellText=metricas,
    colLabels=colunas,
    loc='center'
)

tabela.scale(1,2)

# Histórico do treinamento 
history_dict = history.history # Dicionário com os dados do histórico
history_df = pd.DataFrame(history.history) # Cria DataFrame a partir do histórico
history_df['epoch'] = history_df.index + 1 # Adiciona uma coluna de épocas
history_df = history_df[['epoch'] + [col for col in history_df.columns if col != 'epoch']] # Reorganiza as colunas (opcional, pra deixar epoch no início)

plt.figure(figsize=(15, 4)) # Tamanho gráficos

# 1️ Loss vs Epoch (Treinamento)
plt.subplot(1, 3, 1)
plt.plot(history_dict['loss'], label='Loss (treino)')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 2 Real vs Predito (Treinamento)
plt.subplot(1, 3, 2)
plt.plot(y, y_pred, alpha=0.3)
plt.title('Real vs Predicto')
plt.xlabel('Saída Real (y)')
plt.ylabel('Saída Predita (y_pred)')

plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--',label='Predição Ideal') # Linha de ajuste
plt.legend()

# 3 Erro de Predição
erro =  y - y_pred.flatten() # Calcula o erro de predição (diferença entre valor real e previsto)
plt.subplot(1, 3, 3)
plt.plot(erro)
plt.title('Erro de Predição')
plt.xlabel('Amostra')
plt.ylabel('Erro')
plt.axhline(0, color='m', linestyle='--') # Linha de referência para erro zero

# print(history_df.head()) # Tabela do Histórico
plt.tight_layout()
plt.show()

# Converte o modelo de .tf para .tflite podendo ser embarcado em dispositivos com poucos recursos como ESP32 ou microcontroladores.
print(" -> Convertendo para .tflite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model) # Pega o model (modelo Keras treinado) e prepara-o para a conversão em TFlite.
tflite_model = converter.convert() # Converte o modelo para o formato TensorFlow Lite (.tflite)
with open("modelo_vazao.tflite", "wb") as f:  # Cria o arquivo modelo_vazao.tflite no modo de escrita binária (wb) porque o modelo convertido é binário.
   f.write(tflite_model) # Escreve os bytes do modelo convertido no arquivo OU SEJA está pronto para ser embarcado em dispositivos com poucos recursos.
print(" [OK] Arquivo modelo_vazao.tflite gerado com sucesso!")

