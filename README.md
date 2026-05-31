# SensorVirtual

Projeto desenvolvido como parte do Trabalho de Conclusão de Curso (TCC) em Engenharia Elétrica, com foco na implementação de um sensor virtual para estimativa de vazão utilizando técnicas de aprendizado de máquina e validação em plataforma embarcada ESP32.

## Objetivo

Desenvolver e avaliar modelos de aprendizado de máquina capazes de estimar a vazão de um sistema de aquecimento a partir de variáveis de processo, comparando o desempenho computacional e a viabilidade de implementação em sistemas embarcados.

## Estrutura do Repositório

### Arquivos principais

| Arquivo | Descrição |
|----------|----------|
| `Treino_modelos_real.py` | Treinamento e avaliação dos modelos supervisionados utilizando dados reais do sistema. |
| `RF_completo.py` | Implementação, treinamento e análise do modelo Random Forest. |
| `ProtocoloMQTT.py` | Simulação da comunicação via protocolo MQTT entre o sistema supervisório e o sensor virtual. |
| `ensaio1_dados_reais.csv` | Conjunto de dados utilizado nos experimentos. |

### Implementação embarcada

#### `embarcado_mlp`

Implementação do modelo Multi-Layer Perceptron (MLP) no ESP32.

Arquivos:

- `embarcado_mlp.ino`
- `mlp_model.h`
- `modelo_mlp.tflite`

#### `embarcado_rf`

Implementação do modelo Random Forest no ESP32.

Arquivos:

- `embarcado_rf.ino`
- `rf_model.h`
- `rf_bestmodel.h`

## Tecnologias Utilizadas

- Python
- TensorFlow
- TensorFlow Lite
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- ESP32
- Arduino IDE
- MQTT

## Fluxo Geral do Projeto

1. Coleta dos dados experimentais.
2. Pré-processamento dos dados.
3. Treinamento dos modelos supervisionados.
4. Avaliação dos modelos.
5. Conversão para execução embarcada.
6. Implementação e validação no ESP32.
7. Simulação da comunicação utilizando MQTT.

## Resultados

Os resultados completos, métricas de desempenho e análises comparativas encontram-se descritos no Trabalho de Conclusão de Curso associado a este projeto.

## Autora

Rita Vitória

Curso de Engenharia Elétrica

Centro Federal de Educação Tecnológica de Minas Gerais (CEFET-MG)

## Referência

Caso este repositório seja utilizado como referência, citar:

RITA VITÓRIA. Desenvolvimento e implementação embarcada de um sensor virtual para estimativa de vazão utilizando aprendizado de máquina. Trabalho de Conclusão de Curso, Engenharia Elétrica, CEFET-MG, 2026.

## Licença

Este repositório foi disponibilizado para fins acadêmicos e educacionais.
