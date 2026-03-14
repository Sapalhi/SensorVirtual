# PROJETO FINAL DE CURSO - RITA VITÓRIA
# ORIENTADOR: TÚLIO CHAVES

# SCRIPT PYTHON UTILIZADO PARA PUBLICAÇÃO DE MENSAGENS VIA MQTT

# BIBLIOTECAS
import json
import random
import sys
import time
import pandas as pd
import paho.mqtt.client as mqtt

sys.stdout.reconfigure(encoding='utf-8') # Configura a saída padrão para UTF-8

# DADOS
df_sint = pd.read_csv("dataset_vazao.csv") # Dados sintéticos - João
df_real = pd.read_csv("ensaio1_dados_reais.csv") # Dados Reais - Ensaio 1

usar_dados_reais = True # False = usa df_sint; True = usa df_real

# CONFIGURAÇÃO DO MQTT
    # Funciona APENAS no hotspot do meu celular com minah rede móvel
broker = "node02.myqtthub.com"
port = 1883
username = "Sapalhi08"
password = "q8DhBWv7-O1KlqV1d"
client_id = "literaturasale@gmail.com"
topic = "entrada/vazao"

    # Callback (opcional)
def on_connect(client, userdata, flags, rc): 
    if rc == 0:
        print(" Conectado ao broker MQTT!")
    else:
        print(f" Falha ao conectar, código de erro: {rc}")

    # Conexão 
client = mqtt.Client(client_id=client_id)
client.username_pw_set(username, password)
client.on_connect = on_connect
client.connect(broker, port, keepalive=60)
client.loop_start()

    # Publicação
try:
    total_envios = 10
    for i in range(total_envios):
        
        if usar_dados_reais:
            linha = df_real.sample(1).iloc[0]

            payload = {
                "temperatura": float(linha["Temperature(K)"]),
                "pressao": float(linha["OutputP(Pa)"]),
                "pos_valvula": float(linha["ValveApt(%)"]),
                "vazao_massica": float(linha["Massflow(kg/s)"])
            }

        else:
            linha = df_sint.sample(1).iloc[0]

            payload = {
                "temperatura": float(linha["temperatura"]),
                "pressao": float(linha["pressao"]),
                "pos_valvula": float(linha["posicao_valvula"]),
                "vazao_massica": float(linha["vazao_massica"])
            }

        payload_json = json.dumps(payload)

        result = client.publish(topic, payload_json)
        status = result[0]

        if status == 0:
            print(f" [{i+1}/{total_envios}] JSON publicado: {payload_json}")
        else:
            print(f" Falha na publicacao. Código: {status}")

        time.sleep(0.5)

    print("\n Publicações concluídas com sucesso.")

except Exception as e:
    print(f" Erro durante execucao: {e}")

finally:
    client.loop_stop()
    client.disconnect()
    print(" Desconectado do broker MQTT.")
