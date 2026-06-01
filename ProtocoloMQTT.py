# PROJETO FINAL DE CURSO - RITA VITÓRIA
# SCRIPT PYTHON UTILIZADO PARA PUBLICAÇÃO DE MENSAGENS VIA MQTT

# BIBLIOTECAS
import os
import sys
import time
import json  
import pandas as pd
import paho.mqtt.client as mqtt
import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 

# CONFIGURAÇÕES INICIAIS
sys.stdout.reconfigure(encoding='utf-8') # Configura a saída padrão para UTF-8
connected_flag = False # Variável global para monitorar o status de conexão

dados = []  # Lista para armazenar os dados coletados
NOME_MODELO = "modelo"

BROKER = "<BROKER_IP>"  
PORT = 1883
CLIENT_ID = "<CLIENT_ID>"
    
TOPIC_PUB = "entrada/vazao"
TOPIC_SUB = "saida/vazao"

ARQUIVO_DADOS = "ensaio1_dados_reais.csv" 

TOTAL_AMOSTRAS = 100
DELAY = 0.5  # Segundos entre publicações

# PASTAS PARA SALVAMENTO
PASTA_BASE = r"C:\Users\ritav\SensorVirtual"
PASTA_RESULTADOS = os.path.join(PASTA_BASE, "resultados")
PASTA_GRAFICOS = os.path.join(PASTA_BASE, "graficos")

os.makedirs(PASTA_RESULTADOS, exist_ok=True)
os.makedirs(PASTA_GRAFICOS, exist_ok=True)

ARQUIVO_RESULTADOS = os.path.join(PASTA_RESULTADOS, f"resultados_embarcados.csv")

plt.rcParams.update({'font.size': 16})

# FUNÇÕES
def on_connect(client, userdata, flags, rc): # Callback para monitorar a conexão
    global connected_flag
    print("Código retorno:", rc)
    
    if rc == 0:
        print(" Conectado ao broker MQTT!")
        connected_flag = True
        client.subscribe(TOPIC_SUB) # Incrição no tópico de saída
    else:
        print(f" Falha ao conectar, código de erro: {rc}")

def on_message(client, userdata, msg): # Callback para receber mensagens do tópico de saída
    global dados

    payload = msg.payload.decode()
    print("Recebido:", payload)

    try:
        data = json.loads(payload)
        modelo = data.get("modelo", NOME_MODELO)

        dados.append({
            "modelo": modelo,
            "vazao_real": data["vazao_real"],
            "vazao_estimada": data["vazao_estimada"],
            "erro_abs": data["erro_abs"],
            "erro_pct": data["erro_pct"],
            "tempo_ms": data["tempo_inferencia_ms"]
        })

        print(f"[{modelo}]Coletado {len(dados)}/482 amostras")

    except Exception as e:
        print("Erro JSON:", e)

def conectar_mqtt():
    client = mqtt.Client(client_id=CLIENT_ID, protocol=mqtt.MQTTv311)
    
    client.on_connect = on_connect 
    client.on_message = on_message  
    
    client.connect(BROKER, PORT, keepalive=60)
    client.loop_start()

    while not connected_flag:
        print("Aguardando conexão...")
        time.sleep(1)
    
    print(" Conectado, iniciando envio...")
    return client

def carregar_dados():
    print("Carregando dados...")
    df = pd.read_csv(ARQUIVO_DADOS)
    return df.dropna()

def gerar_payload(linha):
    payload = {
        "temperatura": float(linha["Temperature(K)"]),
        "pressao_entrada": float(linha["IntakeP(Pa)"]),
        "pressao_saida": float(linha["OutputP(Pa)"]),
        "pos_valvula": float(linha["ValveApt(%)"]),
        "vazao_massica": float(linha["Massflow(kg/s)"])
    }
    return json.dumps(payload)

# def publicar_dados(client, df): # De acordo com o númeo de amostras
    
#     for i in range(TOTAL_AMOSTRAS):
#         linha = df.sample(1).iloc[0]
#         payload_json = gerar_payload(linha)
        
#         client.publish(TOPIC_PUB, payload_json)
        
#         print(f"[{i+1}/{TOTAL_AMOSTRAS}] Enviado")
#         time.sleep(DELAY)
        
#     print("\n [OK] Publicações concluídas!")

def publicar_dados(client, df): # Dataset inteiro
    total = len(df)

    for i, linha in df.iterrows():
        payload_json = gerar_payload(linha)
        
        client.publish(TOPIC_PUB, payload_json)
        print(f"[{i+1}/{total}] Enviado")
        
        time.sleep(DELAY)

    print("\n [OK] Publicações concluídas!")

def main():
    try:
        df = carregar_dados()
        client = conectar_mqtt()
        publicar_dados(client, df)
        
        print("\nAguardando respostas...")
        while len(dados) < TOTAL_AMOSTRAS:
            time.sleep(1)
    
    except Exception as e:
        print(f" Erro durante execucao: {e}")
    
    finally:
        client.loop_stop()
        client.disconnect()
        print(" Desconectado do broker MQTT.")

    if len(dados) == 0:
        print("Nenhum dado coletado!")
        return

    df_result = pd.DataFrame(dados)
    modelo_nome = df_result["modelo"].iloc[0]

    # MÉTRICAS DE DESEMPENHO
    r2 = r2_score(df_result["vazao_real"],df_result["vazao_estimada"])
    mae = mean_absolute_error(df_result["vazao_real"],df_result["vazao_estimada"])
    mse = mean_squared_error(df_result["vazao_real"],df_result["vazao_estimada"])

    erro_percentual_medio = df_result["erro_pct"].mean()
    tempo_medio = df_result["tempo_ms"].mean()
    
    # PLANILHAS
    df_metricas = pd.DataFrame([{"Modelo": modelo_nome,"R2": r2,"MAE": mae, "MSE": mse,
    "Erro Percentual Medio (%)": erro_percentual_medio,"Tempo Medio Inferencia (ms)": tempo_medio}])
    
    df_metricas.to_csv(os.path.join(PASTA_RESULTADOS,f"metricas_embarcadas_{modelo_nome}.csv"), index=False)
    print("Planilha de métricas salva!")

    df_result.to_csv(os.path.join(PASTA_RESULTADOS, f"inferencias_detalhadas_embarcado_{modelo_nome}.csv"), index=False)
    print("CSV salvo!")

    # GRÁFICOS
    cor_modelo = "blue" if "MLP" in modelo_nome.upper() else "red"
    # Erro absoluto
    media_erro_abs = df_result["erro_abs"].mean()

    plt.figure(figsize=(7,5), dpi=300)

    plt.hist(df_result["erro_abs"], bins=12, color=cor_modelo, edgecolor="black")

    plt.axvline(media_erro_abs, color='magenta',
        linestyle='--', linewidth=2,
        label=f"Média = {media_erro_abs:.4f}"
    )
    
    plt.xlabel("Erro Absoluto")
    plt.ylabel("Frequência")
    #plt.title(f"Modelo {modelo_nome}: Histograma do Erro Absoluto")

    plt.legend()
    plt.yscale('log')
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter())
    plt.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, f"graficoembarcado_erroabsoluto_{modelo_nome}.png"))
    plt.close()
    
    # Erro percentual
    media_erro = df_result["erro_pct"].mean() # Cálculo da média do erro percentual

    plt.figure(figsize=(7,5), dpi=300)
    plt.hist(df_result["erro_pct"], bins=20, color=cor_modelo, edgecolor="black")
    plt.axvline(media_erro, color='magenta', linestyle='--', linewidth=2, label=f"Média = {media_erro:.2f}%")

    plt.xlabel("Erro Percentual(%)")
    plt.ylabel("Frequência")
    #plt.title(f"Modelo {modelo_nome}: Histograma do Erro Percentual")
    plt.legend()
    plt.yscale('log')
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter())
    plt.grid()
   
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, f"graficoembarcado_erropercentual_{modelo_nome}.png"))
    plt.close()
   
    # Tempo de inferência
    media_tempo = df_result["tempo_ms"].mean()

    plt.figure(figsize=(7,5), dpi=300)
    plt.hist(df_result["tempo_ms"], bins=15, color=cor_modelo, edgecolor="black")
    plt.axvline(media_tempo, color='magenta', linestyle='--', linewidth=2, label=f"Média = {media_tempo:.3f} ms")
             
    plt.xlabel("Tempo de Inferência (ms)")
    plt.ylabel("Frequência")

    plt.legend()
    plt.yscale('log')
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter())
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS,f"graficoembarcado_tempoinferencia_{modelo_nome}.png"))
    plt.close()


# PONTO DE ENTRADA
if __name__ == "__main__":
    main()
    
