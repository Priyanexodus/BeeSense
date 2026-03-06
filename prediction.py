from Model import BeeInference
import json

file = open("Datum\event.json", "r")
data = json.load(file)

engine = BeeInference("./artifacts")
print(engine.predict(data))