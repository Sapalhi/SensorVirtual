#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// -------------------- CONFIGURAÇÕES Wi-Fi --------------------
const char* ssid = "SEU_WIFI";
const char* password = "SENHA_WIFI";

// -------------------- CONFIGURAÇÕES MQTT --------------------
const char* mqtt_server = "node02.myqtthub.com";
const int   mqtt_port = 1883;

const char* mqtt_username = "Sapalhi08";
const char* mqtt_password = "SUA_SENHA_MQTT";

const char* mqtt_topic_sub = "meutopico/entrada";
const char* mqtt_topic_pub = "meutopico/saida";

// -------------------- MQTT CLIENT --------------------
WiFiClient espClient;
PubSubClient client(espClient);

// -------------------- SUA FUNÇÃO CALLBACK --------------------
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Mensagem recebida em ");
  Serial.println(topic);

  String msg = "";
  for (int i = 0; i < length; i++) {
    msg += (char)payload[i];
  }
  Serial.println("Payload: " + msg);
}

// -------------------- WI-FI --------------------
void setup_wifi() {
  delay(10);
  Serial.println("Conectando ao Wi-Fi...");
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(700);
    Serial.print(".");
  }
  Serial.println("\nWi-Fi conectado!");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());
}

// -------------------- RECONNECT MQTT --------------------
void reconnect() {
  while (!client.connected()) {
    Serial.print("Conectando ao broker MQTT... ");
    
    if (client.connect("ESP32Client", mqtt_username, mqtt_password)) {
      Serial.println("Conectado!");

      client.subscribe(mqtt_topic_sub);
      Serial.println("Inscrito no tópico de entrada.");

    } else {
      Serial.print("Falhou. rc=");
      Serial.print(client.state());
      Serial.println(" Tentando novamente em 3 segundos...");
      delay(3000);
    }
  }
}

// -------------------- SETUP PRINCIPAL --------------------
void setup() {
  Serial.begin(115200);
  setup_wifi();

  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(mqttCallback);

  Serial.println("Sistema iniciado!");
}

// -------------------- LOOP PRINCIPAL --------------------
void loop() {

  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  // ------------------ EXEMPLO DE JSON PARA ENVIO ------------------
  StaticJsonDocument<256> doc_out;

  float vazao_real = 12.5;
  float vazao_estimada = 12.1;
  float erro_abs = abs(vazao_real - vazao_estimada);
  float erro_pct = erro_abs / vazao_real * 100.0;
  int tempo_ms = 4;

  doc_out["vazao_real"] = vazao_real;
  doc_out["vazao_estimada"] = vazao_estimada;
  doc_out["erro_abs"] = erro_abs;
  doc_out["erro_pct"] = erro_pct;
  doc_out["tempo_inferencia_ms"] = tempo_ms;

  char payload_out[256];
  serializeJson(doc_out, payload_out, sizeof(payload_out));

  client.publish(mqtt_topic_pub, payload_out);

  Serial.print("Enviado: ");
  Serial.println(payload_out);

  delay(2000);  // ajuste como preferir
}
