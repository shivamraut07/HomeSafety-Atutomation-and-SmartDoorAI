# esp32_simulator.py
import paho.mqtt.client as mqtt
import random
import time

def on_connect(client, userdata, flags, rc):
    print("✅ ESP32 Simulator connected to MQTT")
    
client = mqtt.Client()
client.on_connect = on_connect
client.connect("localhost", 1883, 60)

# Simulate sensor data
while True:
    temp = random.uniform(20.0, 30.0)
    hum = random.uniform(40.0, 80.0)
    gas = random.randint(300, 800)
    motion = random.choice([0, 1])
    window = random.choice([0, 1])
    
    # Publish sensor data
    client.publish("home/sensors/temperature", f"{temp:.1f}")
    client.publish("home/sensors/humidity", f"{hum:.1f}")
    client.publish("home/sensors/gas", str(gas))
    client.publish("home/sensors/motion", str(motion))
    client.publish("home/sensors/window", str(window))
    
    print(f"📤 Sent: Temp={temp:.1f}°C, Hum={hum:.1f}%, Gas={gas}PPM")
    time.sleep(5)