# Deteccion-de-profesiones-en-Twitter
Proyecto de Text Mining / NLP para clasificar tweets menciona al menos una profesión (label=1) o ninguna (label=0). El flujo cubre EDA, preprocesado, fine-tuning de un modelo Transformers (Hugging Face) y generación de predicciones sobre un conjunto de test
Detección de profesiones en Twitter — Proyecto de NLP

El flujo incluye EDA, preprocesado, fine-tuning de un modelo Transformers, evaluación con métricas robustas y exportación de predicciones en formato .tsv.

🎯 Objetivos

Construir un pipeline reproducible de clasificación binaria de texto.

Comparar un baseline clásico con un modelo Transformers y seleccionar el mejor.

Generar un archivo de predicciones estandarizado para uso/evaluación externa.

🗂️ Estructura del repositorio
├─ notebooks/
│  └─ Deteccion_de_profesiones_en_Twitter_GH.ipynb   # cuaderno principal (limpio para GitHub)
├─ src/
│  ├─ eda.py               # análisis exploratorio (opcional)
│  ├─ preprocess.py        # limpieza, normalización, tokenización
│  ├─ train.py             # entrenamiento + validación
│  └─ infer.py             # inferencia + exportación a TSV
├─ data/
│  ├─ raw/                 # datos en bruto (no versionados)
│  └─ processed/           # datos limpios/features (no versionados)
├─ outputs/
│  └─ predicciones.tsv     # archivo de predicciones (no versionado)
├─ figures/                # gráficos (no versionados)
├─ .gitignore
└─ README.md


El repo ignora data/, models/, outputs/ y figures/ para evitar subir datos/artefactos pesados.

🔧 Entorno y dependencias (sugeridas)

Python 3.10+

pandas, numpy, scikit-learn

transformers, datasets, torch (o tensorflow)

matplotlib, seaborn (visualización)

wordcloud (opcional para EDA)

Instalación rápida:

python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -U pip
pip install pandas numpy scikit-learn transformers datasets torch matplotlib seaborn wordcloud

▶️ Uso rápido
1) Datos

Coloca los ficheros en data/raw/ con, al menos:

train.csv → columnas: id, text, label

valid.csv → columnas: id, text, label

test.csv → columnas: id, text (sin label)

Si tu dataset está en otro idioma/estructura, ajusta src/preprocess.py.

2) Entrenamiento

Ejemplo con un modelo multilingüe:

python -m src.train \
  --train data/raw/train.csv \
  --valid data/raw/valid.csv \
  --model_name "bert-base-multilingual-cased" \
  --max_length 128 \
  --batch_size 32 \
  --epochs 3 \
  --lr 2e-5 \
  --out_dir models/best

3) Inferencia y exportación
python -m src.infer \
  --model_dir models/best \
  --test data/raw/test.csv \
  --out outputs/predicciones.tsv


Formato de outputs/predicciones.tsv:

id	label
123	1
456	0
...


Separador: tabulador

Cabeceras: id, label

Sin índice adicional

4) Notebook

También puedes ejecutar el flujo completo desde:

notebooks/Deteccion_de_profesiones_en_Twitter_GH.ipynb


El cuaderno está limpio (sin outputs ni metadatos ruidosos) y documentado paso a paso.

📊 Métricas y validación

Métrica principal sugerida: F1-score (útil con desbalance).

Reporte complementario: Accuracy, Precision, Recall, Matriz de confusión.

Validación con hold-out (train/valid) o K-Fold estratificado.

Semillas fijas para reproducibilidad.

🧠 Modelado (resumen)

Baseline: LogisticRegression / LinearSVC con TF-IDF.

Transformers: fine-tuning (BERT/RoBERTa multilingüe) con tokenización, max_length acorde al tweet y early stopping opcional.

Explicabilidad (opcional): inspección de ejemplos frontera, curvas PR/ROC y análisis de errores.

♻️ Buenas prácticas

Pipelines (scikit-learn / transformers) con parámetros versionados.

Control de data leakage y consistencia de splits.

No versionar datos/artefactos pesados (ya cubierto por .gitignore).

Documentar supuestos del dataset (lengua, hashtags, menciones, emojis).

🚀 Roadmap

Añadir tuning con optuna/GridSearch.

Aumentar robustez con data augmentation para textos cortos.

Publicar un Space (Hugging Face) o endpoint de inferencia liviano.
