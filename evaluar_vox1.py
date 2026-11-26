
# -*- coding: utf-8 -*-
"""
Evaluacion de verificacion de locutor en VoxCeleb1.

Utiliza directamente:
    emb_ecapa.pkl
    emb_wavlm.pkl

Experimentos:
    ECAPA - Protocolo A
    ECAPA - Protocolo B
    WavLM - Protocolo A
    WavLM - Protocolo B

No calcula embeddings ni carga modelos neuronales.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


THRESHOLDS = np.linspace(-1.0, 1.0, 2001)


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluacion VoxCeleb1 usando embeddings PKL."
    )

    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=Path("protocolo_vox1")
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resultados")
    )

    parser.add_argument(
        "--ecapa-pkl",
        type=Path,
        default=Path("emb_ecapa.pkl")
    )

    parser.add_argument(
        "--wavlm-pkl",
        type=Path,
        default=Path("emb_wavlm.pkl")
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["ecapa", "wavlm"],
        default=["ecapa", "wavlm"]
    )

    return parser.parse_args()


# ============================================================
# CARGA DE PROTOCOLOS
# ============================================================

def load_protocols(protocol_dir):

    pairs = pd.read_csv(
        protocol_dir / "veri_test_original.txt",
        sep=r"\s+",
        header=None,
        names=["label", "audio1", "audio2"]
    )

    enrollment = pd.read_csv(
        protocol_dir / "enrollment_10.csv"
    )

    probes = pd.read_csv(
        protocol_dir / "probes.csv"
    )

    trials = pd.read_csv(
        protocol_dir / "trials_registro10.csv"
    )

    required = {
        "enrollment": {"speaker_id", "audio_path"},
        "probes": {"audio_path"},
        "trials": {
            "enrollment_speaker_id",
            "test_audio",
            "label"
        }
    }

    for name, df in [
        ("enrollment", enrollment),
        ("probes", probes),
        ("trials", trials)
    ]:

        missing = required[name] - set(df.columns)

        if missing:
            raise ValueError(
                f"Faltan columnas en {name}: {missing}"
            )

    pairs["label"] = pairs["label"].astype(int)
    trials["label"] = trials["label"].astype(int)

    return pairs, enrollment, probes, trials


def collect_audio_paths(pairs, enrollment, probes, trials):

    paths = set(pairs["audio1"]) | set(pairs["audio2"])

    paths |= set(enrollment["audio_path"])
    paths |= set(probes["audio_path"])
    paths |= set(trials["test_audio"])

    return sorted(str(p) for p in paths)


# ============================================================
# CARGA DE EMBEDDINGS
# ============================================================

def load_pkl(path):

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo PKL: {path.resolve()}"
        )

    print(f"\nCargando: {path}")

    with open(path, "rb") as f:
        embeddings = pickle.load(f)

    if not isinstance(embeddings, dict):
        raise TypeError(
            f"Se esperaba un diccionario en {path}, "
            f"pero se encontro {type(embeddings)}"
        )

    print("Embeddings cargados:", len(embeddings))

    if len(embeddings) > 0:
        print("Ejemplos de claves:")

        for key in list(embeddings.keys())[:3]:
            print(" ", key)

    return embeddings


def normalize_path(path):
    """
    Normaliza rutas para comparar claves de Windows/Linux.
    """

    path = str(path).replace("\\", "/").strip()

    while path.startswith("./"):
        path = path[2:]

    return path.casefold()


def prepare_embeddings(raw_embeddings, audio_paths, model_name):
    """
    Genera un diccionario usando exactamente las rutas del protocolo.

    Permite que las claves del PKL tengan:
      - Separadores / o \\
      - Prefijos adicionales de carpetas
      - Rutas relativas o absolutas

    Si encuentra una coincidencia ambigua, genera un error.
    """

    normalized = {}

    for key, value in raw_embeddings.items():

        norm_key = normalize_path(key)

        normalized.setdefault(norm_key, []).append(value)

    result = {}
    missing = []

    for rel_path in audio_paths:

        target = normalize_path(rel_path)

        # Primero buscar coincidencia exacta.
        candidates = normalized.get(target, [])

        # Si no existe, buscar coincidencias por sufijo.
        if not candidates:

            candidates = [
                value
                for key, values in normalized.items()
                if key.endswith("/" + target)
                or target.endswith("/" + key)
                for value in values
            ]

        if len(candidates) == 0:
            missing.append(rel_path)
            continue

        if len(candidates) > 1:
            raise ValueError(
                f"Coincidencia ambigua para {rel_path} "
                f"en los embeddings de {model_name}. "
                "Revisa las claves del PKL."
            )

        embedding = np.asarray(
            candidates[0],
            dtype=np.float32
        ).reshape(-1)

        if embedding.size == 0 or not np.all(np.isfinite(embedding)):
            raise ValueError(
                f"Embedding invalido para {rel_path} ({model_name})"
            )

        result[rel_path] = embedding

    print(f"\n{model_name.upper()}:")
    print("Embeddings encontrados:", len(result))
    print("Embeddings faltantes:", len(missing))

    if missing:
        print("\nPrimeras rutas sin embedding:")

        for path in missing[:10]:
            print(" ", path)

        raise KeyError(
            f"Faltan {len(missing)} embeddings de {model_name}. "
            "Revisa las claves del archivo PKL."
        )

    # Comprobar que todos tengan la misma dimension.
    dimensions = {
        vector.shape[0] for vector in result.values()
    }

    if len(dimensions) != 1:
        raise ValueError(
            f"Dimensiones inconsistentes en {model_name}: {dimensions}"
        )

    print("Dimension de embedding:", next(iter(dimensions)))

    return result


# ============================================================
# SIMILITUD COSENO
# ============================================================

def cosine(a, b):

    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    if a.shape != b.shape:
        raise ValueError(
            f"Dimensiones incompatibles: {a.shape} y {b.shape}"
        )

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


# ============================================================
# PROTOCOLO A
# ============================================================

def scores_protocol_a(embeddings, pairs):

    scores = []

    for audio1, audio2 in zip(
        pairs["audio1"],
        pairs["audio2"]
    ):

        score = cosine(
            embeddings[audio1],
            embeddings[audio2]
        )

        scores.append(score)

    return np.asarray(scores)


# ============================================================
# PROTOCOLO B
# ============================================================

def enrollment_templates(embeddings, enrollment):

    templates = {}

    for speaker_id, group in enrollment.groupby("speaker_id"):

        vectors = [
            embeddings[path]
            for path in group["audio_path"]
        ]

        # Promedio de los embeddings de registro.
        template = np.mean(
            np.stack(vectors),
            axis=0
        )

        # Normalizar el template.
        norm = np.linalg.norm(template)

        if norm > 0:
            template = template / norm

        templates[speaker_id] = template

    return templates


def scores_protocol_b(embeddings, enrollment, trials):

    templates = enrollment_templates(
        embeddings,
        enrollment
    )

    scores = []

    for speaker_id, test_audio in zip(
        trials["enrollment_speaker_id"],
        trials["test_audio"]
    ):

        score = cosine(
            templates[speaker_id],
            embeddings[test_audio]
        )

        scores.append(score)

    return np.asarray(scores)


# ============================================================
# METRICAS FAR / FRR / EER
# ============================================================

def calculate_far_frr(scores, labels):

    genuine = scores[labels == 1]
    impostor = scores[labels == 0]

    if len(genuine) == 0 or len(impostor) == 0:
        raise ValueError(
            "Se necesitan trials genuinos e impostores."
        )

    far = np.asarray([
        np.mean(impostor >= threshold)
        for threshold in THRESHOLDS
    ])

    frr = np.asarray([
        np.mean(genuine < threshold)
        for threshold in THRESHOLDS
    ])

    # EER mediante el punto mas cercano.
    idx = int(np.argmin(np.abs(far - frr)))

    eer = float((far[idx] + frr[idx]) / 2)
    eer_threshold = float(THRESHOLDS[idx])

    # Interpolacion si FAR y FRR se cruzan.
    diff = far - frr

    crossings = np.where(
        np.sign(diff[:-1]) != np.sign(diff[1:])
    )[0]

    if len(crossings):

        j = int(crossings[0])

        x0 = THRESHOLDS[j]
        x1 = THRESHOLDS[j + 1]

        d0 = diff[j]
        d1 = diff[j + 1]

        if d1 != d0:

            threshold_interp = float(
                x0 - d0 * (x1 - x0) / (d1 - d0)
            )

            far_interp = float(
                np.interp(threshold_interp, THRESHOLDS, far)
            )

            frr_interp = float(
                np.interp(threshold_interp, THRESHOLDS, frr)
            )

            eer = (far_interp + frr_interp) / 2
            eer_threshold = threshold_interp

    return far, frr, eer, eer_threshold


# ============================================================
# GRAFICOS
# ============================================================

def plot_score_histograms(scores_dict, labels_dict, output_dir):

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, (name, scores) in zip(
        axes.flatten(),
        scores_dict.items()
    ):

        protocol = "A" if name.endswith("- A") else "B"
        labels = labels_dict[protocol]

        ax.hist(
            scores[labels == 1],
            bins=100,
            density=True,
            alpha=0.6,
            label="Genuinos"
        )

        ax.hist(
            scores[labels == 0],
            bins=100,
            density=True,
            alpha=0.6,
            label="Impostores"
        )

        ax.set_title(name)
        ax.set_xlabel("Similitud coseno")
        ax.set_ylabel("Densidad")
        ax.legend()

    fig.tight_layout()

    path = output_dir / "histogramas.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print("Grafico guardado:", path)


def plot_far_frr(results, output_dir):

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, (name, values) in zip(
        axes.flatten(),
        results.items()
    ):

        far, frr, eer, threshold = values

        ax.plot(THRESHOLDS, 100 * far, label="FAR")
        ax.plot(THRESHOLDS, 100 * frr, label="FRR")

        ax.plot(
            threshold,
            100 * eer,
            "ko",
            label=f"EER = {100 * eer:.2f}%"
        )

        ax.set_title(name)
        ax.set_xlabel("Umbral")
        ax.set_ylabel("Tasa de error (%)")
        ax.legend()

    fig.tight_layout()

    path = output_dir / "far_frr.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print("Grafico guardado:", path)


def plot_roc(results, output_dir):

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, protocol in zip(axes, ["A", "B"]):

        for model_name in ["ECAPA", "WavLM"]:

            name = f"{model_name} - {protocol}"

            if name not in results:
                continue

            far, frr, eer, _ = results[name]

            ax.plot(
                100 * far,
                100 * (1 - frr),
                label=f"{model_name} (EER = {100 * eer:.2f}%)"
            )

        ax.set_xscale("log")
        ax.set_xlim(0.01, 100)
        ax.set_ylim(0, 100)

        ax.set_title(f"Curva ROC - Protocolo {protocol}")
        ax.set_xlabel("FAR (%)")
        ax.set_ylabel("Tasa de aceptacion genuina (%)")

        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.tight_layout()

    path = output_dir / "roc.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print("Grafico guardado:", path)


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Cargar protocolos
    # --------------------------------------------------------

    pairs, enrollment, probes, trials = load_protocols(
        args.protocol_dir
    )

    print("\nProtocolos cargados:")
    print("Pares protocolo A:", len(pairs))
    print(
        "Audios de registro:", len(enrollment),
        "| personas:", enrollment["speaker_id"].nunique()
    )
    print("Audios de prueba:", len(probes))
    print("Comparaciones protocolo B:", len(trials))

    print("A ->", pairs["label"].value_counts().to_dict())
    print("B ->", trials["label"].value_counts().to_dict())

    audio_paths = collect_audio_paths(
        pairs,
        enrollment,
        probes,
        trials
    )

    print("\nAudios distintos utilizados:", len(audio_paths))

    # --------------------------------------------------------
    # Cargar PKL
    # --------------------------------------------------------

    labels_a = pairs["label"].to_numpy()
    labels_b = trials["label"].to_numpy()

    labels_dict = {
        "A": labels_a,
        "B": labels_b
    }

    all_scores = {}

    for model_name in args.models:

        if model_name == "ecapa":
            pkl_path = args.ecapa_pkl
        else:
            pkl_path = args.wavlm_pkl

        # Cargar embeddings originales.
        raw_embeddings = load_pkl(pkl_path)

        # Adaptar claves a las rutas del protocolo.
        embeddings = prepare_embeddings(
            raw_embeddings,
            audio_paths,
            model_name
        )

        # ----------------------------------------------------
        # Protocolo A
        # ----------------------------------------------------

        scores_a = scores_protocol_a(
            embeddings,
            pairs
        )

        all_scores[f"{model_name.upper()} - A"] = scores_a

        # ----------------------------------------------------
        # Protocolo B
        # ----------------------------------------------------

        scores_b = scores_protocol_b(
            embeddings,
            enrollment,
            trials
        )

        all_scores[f"{model_name.upper()} - B"] = scores_b

        print(f"\n{model_name.upper()} evaluado correctamente.")

        # Liberar memoria del modelo y sus embeddings.
        del embeddings
        del raw_embeddings

    # --------------------------------------------------------
    # Guardar puntajes
    # --------------------------------------------------------

    scores_path = args.output_dir / "puntajes.pkl"

    with open(scores_path, "wb") as f:
        pickle.dump(all_scores, f)

    print("\nPuntajes guardados:", scores_path)

    # --------------------------------------------------------
    # Calcular metricas
    # --------------------------------------------------------

    results = {}
    summary = []

    for name, scores in all_scores.items():

        protocol = "A" if name.endswith("- A") else "B"
        labels = labels_dict[protocol]

        genuine = scores[labels == 1]
        impostor = scores[labels == 0]

        far, frr, eer, threshold = calculate_far_frr(
            scores,
            labels
        )

        results[name] = (
            far,
            frr,
            eer,
            threshold
        )

        print(
            f"\n{name} | n={len(scores)}"
            f"\n  Genuinos promedio   = {genuine.mean():.4f}"
            f"\n  Impostores promedio = {impostor.mean():.4f}"
            f"\n  EER                 = {100 * eer:.2f}%"
            f"\n  Umbral EER          = {threshold:.4f}"
        )

        summary.append({
            "modelo_protocolo": name,
            "n_trials": len(scores),
            "genuinos_promedio": float(genuine.mean()),
            "impostores_promedio": float(impostor.mean()),
            "EER": eer,
            "EER_porcentaje": 100 * eer,
            "umbral_EER": threshold
        })

    # --------------------------------------------------------
    # Guardar resumen CSV
    # --------------------------------------------------------

    summary_df = pd.DataFrame(summary)

    csv_path = args.output_dir / "metricas.csv"

    summary_df.to_csv(
        csv_path,
        index=False
    )

    print("\nMetricas guardadas:", csv_path)

    # --------------------------------------------------------
    # Graficos
    # --------------------------------------------------------

    plot_score_histograms(
        all_scores,
        labels_dict,
        args.output_dir
    )

    plot_far_frr(
        results,
        args.output_dir
    )

    plot_roc(
        results,
        args.output_dir
    )

    print("\nEvaluacion terminada.")
    print("Resultados en:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
