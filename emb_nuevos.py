from pathlib import Path
import pickle
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_AUDIO = Path("vox1_test_wav/wav")
PROTOCOL_DIR = Path("protocolo_vox1")
RESULTS_DIR = Path("resultados")

N_CASES = 5

# ============================================================
# 1. CARGAR PUNTAJES
# ============================================================

with open(RESULTS_DIR / "puntajes.pkl", "rb") as f:
    puntajes = pickle.load(f)

print("Puntajes disponibles:", list(puntajes.keys()))

# ============================================================
# 2. CARGAR PROTOCOLO A
# ============================================================

pairs = pd.read_csv(
    PROTOCOL_DIR / "veri_test_original.txt",
    sep=r"\s+",
    header=None,
    names=["label", "audio1", "audio2"]
)

pairs["speaker1"] = pairs["audio1"].apply(
    lambda x: Path(x).parts[0]
)

pairs["speaker2"] = pairs["audio2"].apply(
    lambda x: Path(x).parts[0]
)

# ============================================================
# 3. CARGAR PROTOCOLO B
# ============================================================

enrollment = pd.read_csv(
    PROTOCOL_DIR / "enrollment_10.csv"
)

trials = pd.read_csv(
    PROTOCOL_DIR / "trials_registro10.csv"
)

# Normalizar identificadores
enrollment["speaker_id"] = enrollment["speaker_id"].astype(str)
trials["enrollment_speaker_id"] = (
    trials["enrollment_speaker_id"].astype(str)
)
trials["test_speaker_id"] = trials["test_speaker_id"].astype(str)

# ============================================================
# 4. CARGAR UMBRALES
# ============================================================

metricas = pd.read_csv(RESULTS_DIR / "metricas.csv")

print("\nColumnas de metricas.csv:")
print(metricas.columns.tolist())

# Construir diccionario de umbrales
thresholds = {}

for _, row in metricas.iterrows():
    key = str(row["modelo_protocolo"])
    thresholds[key] = float(row["umbral_EER"])

print("\nUmbrales:")
print(thresholds)

# ============================================================
# 5. VALIDAR CONSISTENCIA
# ============================================================

# Protocolo A: etiqueta y hablantes
pairs["label_correct"] = (
    (pairs["speaker1"] == pairs["speaker2"])
    == (pairs["label"] == 1)
)

print("\n--- PROTOCOLO A ---")
print("Número de pares:", len(pairs))
print("Etiquetas incorrectas:", (~pairs["label_correct"]).sum())

# Protocolo B: etiqueta y hablantes
trials["label_correct"] = (
    (trials["enrollment_speaker_id"] == trials["test_speaker_id"])
    == (trials["label"] == 1)
)

print("\n--- PROTOCOLO B ---")
print("Número de trials:", len(trials))
print("Etiquetas incorrectas:", (~trials["label_correct"]).sum())

# Verificar longitudes de puntajes
for model in ["ECAPA", "WAVLM"]:
    for protocol, expected in [
        ("A", len(pairs)),
        ("B", len(trials))
    ]:
        key = f"{model} - {protocol}"

        if key not in puntajes:
            raise KeyError(f"No existe la clave {key} en puntajes.pkl")

        scores = np.asarray(puntajes[key]).reshape(-1)

        print(
            f"{key}: {len(scores)} puntajes "
            f"(esperados: {expected})"
        )

        if len(scores) != expected:
            raise ValueError(
                f"ERROR: {key} tiene {len(scores)} puntajes, "
                f"pero el protocolo tiene {expected} filas."
            )

# Detener si las etiquetas no coinciden
if not pairs["label_correct"].all():
    raise ValueError("Hay etiquetas inconsistentes en el protocolo A.")

if not trials["label_correct"].all():
    raise ValueError("Hay etiquetas inconsistentes en el protocolo B.")

print("\nValidaciones superadas.")