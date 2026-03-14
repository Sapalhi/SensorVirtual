# REPRODUÇÃO DO JOÃO GABRIEL DRUMMOND
# SCRIPT PYTHON UTILIZADO PARA GERAÇÃO DOS DADOS DO TREINAMENTO

# Bibliotecas
import numpy as np
import pandas as pd

# Parâmetros -----------------------------------------------------------------------------------------------------
NUM_AMOSTRAS = 10000 # Número de amostras
CD = 0.7 # Coeficiente de descarga da valvula (adimensional)
DIAMETRO_VV_MM = 20 # Diâmetro da valvula (mm)
R = 287.058 # Constante dos gases para o ar (J/kg·K)

TEMP_MIN, TEMP_MAX = 20, 30 # Temperatura (°C)
PRESSAO_RES_MIN, PRESSAO_RES_MAX = 1.2, 8 # Pressao do reservatório (bar)
PRESSAO_AMBIENTE_BAR = 1.0 # Pressão ambiente (bar)
DIAMETRO_M = DIAMETRO_VV_MM / 1000 # Conversão de mm -> m
AREA_MAX = np.pi * (DIAMETRO_M / 2) ** 2 #  ́Area da válvula (m²)
PRESSAO_AMBIENTE_PA = PRESSAO_AMBIENTE_BAR * 1e5 # Conversão de bar -> Pa

# Operações -----------------------------------------------------------------------------------------------------
temperatura_C = np.random.uniform(TEMP_MIN, TEMP_MAX, size=NUM_AMOSTRAS) # Amostra de temperaturas (°C)
temperatura_K = temperatura_C + 273.15

pressao_res_bar = np.random.uniform(PRESSAO_RES_MIN, PRESSAO_RES_MAX, size=NUM_AMOSTRAS) # Amostra de pressões do reservatório (bar)
pressao_res_Pa = pressao_res_bar * 1e5

deltaP = pressao_res_Pa - PRESSAO_AMBIENTE_PA # Diferença de pressão (Pa) entre reservatório e ambiente

abertura_normalizada = np.random.uniform(0, 1, size=NUM_AMOSTRAS) # Grau de abertura normalizado da válvula (0 a 1)

# Cálculo da vazão mássica (kg/s) utilizando a fórmula fornecida
vazao_massica = (pressao_res_Pa/(R*temperatura_K)) * CD * AREA_MAX * abertura_normalizada * np.sqrt((2 * deltaP * R * temperatura_K) / pressao_res_Pa)

# Salvamento dos dados em CSV -----------------------------------------------------------------------------------
df = pd.DataFrame({
'temperatura': np.round(temperatura_C, 4), 
'pressao': np.round(pressao_res_bar, 4),
'posicao_valvula': np.round(abertura_normalizada, 4),
'vazao_massica': np.round(vazao_massica, 4)
})
df.to_csv('dataset_vazao.csv', index=False)

