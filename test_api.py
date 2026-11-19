import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import tensorflow as tf

API_URL = "http://127.0.0.1:8000/predict"
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

PROJECT_DIR = Path(__file__).resolve().parent
(_, _), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

results = []
for index in range(20):
    payload = {"pixels": test_images[index].flatten().tolist()}
    response = requests.post(API_URL, json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()

    print("Réponse JSON :", result)

    real_id = int(test_labels[index])
    is_correct = real_id == result["class_id"]
    row = {
        "Image": index,
        "Classe réelle": CLASS_NAMES[real_id],
        "Classe prédite": result["class_name"],
        "Confiance": result["confidence"],
        "Latence (ms)": result["latency_ms"],
        "Résultat": "Correct" if is_correct else "Incorrect",
    }
    results.append(row)
    print(row)

correct = sum(row["Résultat"] == "Correct" for row in results)
mean_latency = float(np.mean([row["Latence (ms)"] for row in results]))
print(f"\nPrédictions correctes : {correct}/20")
print(f"Taux de réussite : {correct / 20 * 100:.2f} %")
print(f"Latence moyenne : {mean_latency:.2f} ms")

with (PROJECT_DIR / "resultats_20_images.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

plt.imshow(test_images[0], cmap="gray")
plt.title(f"Réelle : {results[0]['Classe réelle']} | Prédite : {results[0]['Classe prédite']}")
plt.axis("off")
plt.tight_layout()
plt.savefig(PROJECT_DIR / "image_testee.png", dpi=160)
plt.show()

bad_response = requests.post(API_URL, json={"pixels": [1, 2, 3]}, timeout=30)
print("\nTest incorrect - HTTP", bad_response.status_code)
print(bad_response.json())
assert bad_response.status_code == 400
assert bad_response.json()["detail"] == "L'entrée doit contenir exactement 784 pixels."
