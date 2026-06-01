//  BIBLIOTECAS

#include <WiFi.h>  // Conecta o ESP à rede Wifi
#include <PubSubClient.h> // Implementa o protocolo MQTT
#include <ArduinoJson.h>  // Manipulação de JSON (ler e criar mensagens)
#include <Arduino.h>  // base do Arduino (funções padrões)
#include <ArduTFLite.h>  // Permite modelo rodar no ESP
#include "mlp_model.h"  // Contém modelo convertido (pesos + estrutura)

// VARIÁVEIS
// WIFI
const char* ssid = "<nome_do_wifi>";
const char* password = "<senha_do_wifi>";

// MQTT
const char* mqtt_server = "<servidor_do_mqtt>";
const int mqtt_port = 1883;
const char* client_id = "<id_cliente>";
const char* mqtt_topic_sub = "entrada/vazao"; // SUB = Recebe dados
const char* mqtt_topic_pub = "saida/vazao"; // PUB = envia/publica dados

constexpr int kTensorArenaSize = 32 * 1024;   // Reserva 32KB de RAM para TensorFlow Lite
uint8_t tensor_arena[kTensorArenaSize];

// NORMALIZAÇÃO
float valvula_min = 9.2, valvula_max = 17.6;
float temp_min = 294.45, temp_max = 296.95;
float pressaoint_min = 631813.99, pressaoint_max = 1001156.15;
float pressaoext_min = 88366.87, pressaoext_max = 95477.35;

WiFiClient espClient;
PubSubClient client(espClient);

float normalizar(float valor, float min, float max) { 
  return (valor - min) / (max - min);
}

  // Função de reconexão do MQTT
void reconnect() {
  while (!client.connected()) {
    Serial.println(" Tentando conectar ao MQTT...");
    if (client.connect(client_id)) {
      Serial.println(" Conectado ao MQTT!");
      client.subscribe(mqtt_topic_sub); // Começa a "escutar" o tópico
    } else {
      Serial.print(" Falha, rc=");
      Serial.print(client.state());
      delay(5000); // Intervalo de 5s para a próxima tentativa de conexão
    }
  }
}

// ------------------------- CHAMADA AUTOMÁTICA (CALLBACK) --------------------------------
  // Ativada quando chega mensagem no MQTT
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  Serial.println("CALLBACK CHAMADO");
  unsigned long tempo_inicio_inferencia = micros(); //  Tempo de Inferência
  // Conversão de bytes em string
  char msg[length + 1];
  memcpy(msg, payload, length);
  msg[length] = '\0';
  // Conversão de string JSON em objeto
  StaticJsonDocument<256> doc;
  DeserializationError error = deserializeJson(doc, msg);
  if (error) {
    Serial.print(" Erro ao interpretar JSON: ");
    Serial.println(error.c_str());
    return;
  }
  // Garantia que todos os dados existem
  if (!doc.containsKey("temperatura") || !doc.containsKey("pressao_entrada") || !doc.containsKey("pressao_saida") || !doc.containsKey("pos_valvula") 
      || !doc.containsKey("vazao_massica")) {
          Serial.println(" Campos necessarios nao encontrados no JSON");
          return;
  }

  // ------------------------- EXTRAÇÃO DE VARIÁVEIS --------------------------
  float temperatura = doc["temperatura"];
  float pressao_entrada = doc["pressao_entrada"];
  float pressao_saida = doc["pressao_saida"];
  float posicao_valvula = doc["pos_valvula"];
  float vazao_real = doc["vazao_massica"];
    // Normalização dos dados de entrada
  float temp_norm = normalizar(temperatura, temp_min, temp_max);
  float press_in_norm = normalizar(pressao_entrada, pressaoint_min, pressaoint_max);
  float press_out_norm = normalizar(pressao_saida, pressaoext_min, pressaoext_max);
  float valv_norm = normalizar(posicao_valvula, valvula_min, valvula_max);

  // ------------------------- INFERÊNCIA -----------------------------------
    // Define entradas do modelo
  modelSetInput(valv_norm, 0);
  modelSetInput(temp_norm, 1);
  modelSetInput(press_in_norm, 2);
  modelSetInput(press_out_norm, 3);
    // Executa o modelo
  if (!modelRunInference()) {
    Serial.println(" Falha na inferência!");
    return;
  }
  float vazao_estimada = modelGetOutput(0); // Pega saída (previsão)
  unsigned long tempo_fim_inferencia = micros();
    // Tempo
  float tempo_ms = (tempo_fim_inferencia - tempo_inicio_inferencia) / 1000.0;

  // ------------------------- CÁLCULO DE ERRO --------------------------------
  float erro_abs = abs(vazao_real - vazao_estimada);
    // Evitar divisão por zero (IMPORTANTE)
  float erro_pct = 0.0;
  if (vazao_real != 0) {
    erro_pct = (erro_abs / vazao_real) * 100.0; // Erro percentual = MAPE
  }

  Serial.println("----------------------------------");
  Serial.printf("Temperatura: %.2f | Valvula: %.2f\n", temperatura, posicao_valvula);
  Serial.printf("Pressao Entrada: %.2f | Pressao Saida: %.2f\n", pressao_entrada, pressao_saida);
  Serial.printf(">> Vazao real: %.6f\n", vazao_real);
  Serial.printf(">> Vazao estimada: %.6f\n", vazao_estimada);
  Serial.printf("Erro absoluto: %.6f | Erro percentual (MAPE): %.2f%%\n", erro_abs, erro_pct);
  Serial.printf("Tempo de inferencia: %.6f ms\n", tempo_ms);
  Serial.println("----------------------------------\n");

  // ----------------- PUBLICAÇÃO DS RESULTADOS VIA MQTT ---------------------------
  StaticJsonDocument<256> doc_out;
    // Modelo
  doc_out["modelo"] = "MLP";
    // Entradas (opcional, mas MUITO bom pra debug e TCC)
  doc_out["temperatura"] = temperatura;
  doc_out["pos_valvula"] = posicao_valvula;
  doc_out["pressao_entrada"] = pressao_entrada;
  doc_out["pressao_saida"] = pressao_saida;
    // Saídas
  doc_out["vazao_real"] = vazao_real;
  doc_out["vazao_estimada"] = vazao_estimada;
    // Métricas
  doc_out["erro_abs"] = erro_abs;
  doc_out["erro_pct"] = erro_pct;
  doc_out["tempo_inferencia_ms"] = tempo_ms;
  char payload_out[256];
  serializeJson(doc_out, payload_out, sizeof(payload_out)); // Serializa
  client.publish(mqtt_topic_pub, payload_out);  // Publicação
}

void setup_wifi() {
  delay(10);
  Serial.print("Conectando ao Wi-Fi ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }
  Serial.println("\n Conectado ao Wi-Fi");
}

void setup() {
  Serial.begin(115200);
  while (!Serial);
  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(mqttCallback);
  Serial.println(" Inicializando modelo...");
    if (!modelInit(model, tensor_arena, kTensorArenaSize)) { // Carrega o modelo na memória
      Serial.println(" Erro ao carregar o modelo!");
      while (true);
    }
  Serial.println(" Modelo carregado com sucesso!");
  Serial.println(" Aguardando dados no t ́opico MQTT...");
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();  // Processa mensagens MQTT
}
