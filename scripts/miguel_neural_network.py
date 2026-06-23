import argparse
import json
import os
import pickle
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, Model


IMG_SIZE = (224, 224)

KAGGLE_DATASET = "miguel7penteado/PCS5022-2026"
DEFAULT_MODEL_DIR_NAME = "model_logic_vqa"


# ---------------------------------------------------------------------
# Ambiente e caminhos
# ---------------------------------------------------------------------

def running_in_colab():
    """
    Retorna True quando o script está sendo executado no Google Colab.
    """
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def get_project_root(args_root=None):
    """
    Define a raiz do projeto.

    Cenario local:
        projeto/
        - dataset/
        - scripts/
        - src/
        - notebooks/
        - tests/

    Se este arquivo estiver em scripts/, a raiz sera Path(__file__).parents[1].

    Cenario Colab:
        recomenda-se usar --project-root /content/PCS5022_2026
        ou deixar o padrao /content/PCS5022_2026.
    """
    if args_root:
        return Path(args_root).expanduser().resolve()

    if running_in_colab():
        return Path("/content/PCS5022_2026").resolve()

    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        return Path.cwd().resolve()


def get_dataset_root(project_root):
    return project_root / "dataset"


def get_train_images_dir(project_root):
    return get_dataset_root(project_root) / "train_dataset" / "images"


def get_train_text_dir(project_root):
    return get_dataset_root(project_root) / "train_dataset" / "text"


def get_test_images_dir(project_root):
    return get_dataset_root(project_root) / "test_dataset" / "images"


def get_test_text_dir(project_root):
    return get_dataset_root(project_root) / "test_dataset" / "text"


def get_model_dir(project_root, model_dir=None):
    if model_dir:
        p = Path(model_dir)
        if p.is_absolute():
            return p
        return project_root / p
    return project_root / DEFAULT_MODEL_DIR_NAME


def find_single_jsonl(text_dir, preferred_names=None):
    """
    Encontra um arquivo JSONL dentro de uma pasta text/.
    """
    text_dir = Path(text_dir)

    if preferred_names:
        for name in preferred_names:
            candidate = text_dir / name
            if candidate.exists():
                return candidate

    jsonl_files = sorted(text_dir.glob("*.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError(f"Nenhum arquivo .jsonl encontrado em: {text_dir}")

    if len(jsonl_files) > 1:
        print(f"Aviso: mais de um JSONL encontrado em {text_dir}. Usando: {jsonl_files[0]}")
        print("Arquivos encontrados:")
        for p in jsonl_files:
            print(f"  - {p}")

    return jsonl_files[0]


def resolve_default_paths(project_root):
    """
    Resolve automaticamente os caminhos esperados para treino e teste.
    """
    train_jsonl = find_single_jsonl(
        get_train_text_dir(project_root),
        preferred_names=["perguntas.jsonl", "train.jsonl", "questions_train.jsonl"],
    )

    test_jsonl = find_single_jsonl(
        get_test_text_dir(project_root),
        preferred_names=["perguntas.jsonl", "test.jsonl", "questions_test.jsonl"],
    )

    return {
        "train_jsonl": train_jsonl,
        "train_images_dir": get_train_images_dir(project_root),
        "test_jsonl": test_jsonl,
        "test_images_dir": get_test_images_dir(project_root),
    }


def print_environment_summary(project_root):
    print("Resumo do ambiente")
    print("------------------")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Sistema: {platform.platform()}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"Executando no Colab: {running_in_colab()}")
    print(f"Raiz do projeto: {project_root}")
    print(f"Dataset esperado em: {get_dataset_root(project_root)}")
    print()


# ---------------------------------------------------------------------
# Kaggle
# ---------------------------------------------------------------------

def ensure_kaggle_cli():
    """
    Garante que o pacote kaggle esteja instalado.
    """
    try:
        import kaggle  # noqa: F401
        return
    except Exception:
        pass

    print("Pacote kaggle nao encontrado. Instalando...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "kaggle"],
        check=True,
    )


def configure_kaggle_credentials(kaggle_json=None):
    """
    Configura o arquivo kaggle.json.
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)

    target = kaggle_dir / "kaggle.json"

    if kaggle_json:
        source = Path(kaggle_json).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Arquivo kaggle.json nao encontrado: {source}")
        shutil.copy2(source, target)

    if not target.exists():
        raise FileNotFoundError(
            "Credenciais Kaggle nao encontradas.\n"
            "Crie o token no Kaggle em Account > Create New API Token.\n"
            f"Depois salve o arquivo em: {target}\n"
            "Ou execute este script com: --kaggle-json caminho\\para\\kaggle.json"
        )

    try:
        os.chmod(target, 0o600)
    except Exception:
        pass

    print(f"Credenciais Kaggle configuradas em: {target}")


def download_kaggle_dataset(project_root, kaggle_json=None, force=False):
    """
    Baixa o dataset do Kaggle para a pasta dataset/ da raiz do projeto.
    """
    ensure_kaggle_cli()
    configure_kaggle_credentials(kaggle_json)

    dataset_root = get_dataset_root(project_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    expected_dirs = [
        dataset_root / "train_dataset" / "images",
        dataset_root / "train_dataset" / "text",
        dataset_root / "test_dataset" / "images",
        dataset_root / "test_dataset" / "text",
    ]

    if all(p.exists() for p in expected_dirs) and not force:
        print("Dataset ja parece estar baixado. Use --force para baixar novamente.")
        return

    print(f"Baixando dataset Kaggle: {KAGGLE_DATASET}")
    print(f"Destino: {dataset_root}")

    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET,
            "-p",
            str(dataset_root),
            "--unzip",
        ],
        check=True,
    )

    print("Download concluido.")
    validate_dataset_structure(project_root)


def validate_dataset_structure(project_root):
    """
    Verifica se a estrutura esperada existe.
    """
    required_dirs = [
        get_dataset_root(project_root),
        get_train_images_dir(project_root),
        get_train_text_dir(project_root),
        get_test_images_dir(project_root),
        get_test_text_dir(project_root),
    ]

    for p in required_dirs:
        if not p.exists():
            raise FileNotFoundError(f"Pasta obrigatoria nao encontrada: {p}")

    train_png = len(list(get_train_images_dir(project_root).glob("*.png")))
    test_png = len(list(get_test_images_dir(project_root).glob("*.png")))
    train_jsonl = len(list(get_train_text_dir(project_root).glob("*.jsonl")))
    test_jsonl = len(list(get_test_text_dir(project_root).glob("*.jsonl")))

    print()
    print("Estrutura do dataset")
    print("--------------------")
    print(f"Imagens de treino: {train_png}")
    print(f"JSONL de treino: {train_jsonl}")
    print(f"Imagens de teste: {test_png}")
    print(f"JSONL de teste: {test_jsonl}")
    print()

    if train_png == 0:
        print("Aviso: nenhuma imagem PNG encontrada em train_dataset/images.")
    if test_png == 0:
        print("Aviso: nenhuma imagem PNG encontrada em test_dataset/images.")
    if train_jsonl == 0:
        print("Aviso: nenhum JSONL encontrado em train_dataset/text.")
    if test_jsonl == 0:
        print("Aviso: nenhum JSONL encontrado em test_dataset/text.")


# ---------------------------------------------------------------------
# Leitura de dados
# ---------------------------------------------------------------------

def read_jsonl(path, require_answer=True):
    rows = []
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                row = json.loads(line)

                if require_answer and "answer" not in row:
                    raise ValueError(
                        f"Linha {line_number} sem campo 'answer' em {path}: {row}"
                    )

                rows.append(row)

    df = pd.DataFrame(rows)

    required_columns = ["index", "question"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatoria ausente no JSONL {path}: {col}")

    return df


def find_image_path(images_dir, index):
    """
    Aceita nomes comuns:
    0000.png, 0.png, circuit_0.png, page-0001.png, page_0001.png, image_0000.png.
    """
    images_dir = Path(images_dir)

    candidates = [
        images_dir / f"{index:04d}.png",
        images_dir / f"{index}.png",
        images_dir / f"circuit_{index}.png",
        images_dir / f"circuit_{index:04d}.png",
        images_dir / f"image_{index:04d}.png",
        images_dir / f"img_{index:04d}.png",
        images_dir / f"page-{index + 1:04d}.png",
        images_dir / f"page_{index + 1:04d}.png",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Imagem do indice {index} nao encontrada em {images_dir}. "
        f"Tente nomear como {index:04d}.png."
    )


def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def normalize_question(text):
    text = str(text).strip().lower()
    text = text.replace("=", " = ")
    text = text.replace(",", " , ")
    text = re.sub(r"\s+", " ", text)
    return text


def build_text_vectorizer(train_texts, max_tokens=4000, seq_len=80):
    vectorizer = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=seq_len,
        standardize=None,
    )
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_texts).batch(64))
    return vectorizer


def make_dataset(df, images_dir, vectorizer, label_encoder=None, batch_size=32, shuffle=False):
    image_paths = [str(find_image_path(images_dir, int(i))) for i in df["index"]]
    questions = [normalize_question(q) for q in df["question"].astype(str)]

    if "answer" in df.columns and label_encoder is not None:
        y = label_encoder.transform(df["answer"].astype(str))
    else:
        y = None

    seq_len = int(vectorizer.get_config()["output_sequence_length"])

    def gen():
        for n, path in enumerate(image_paths):
            image = load_image(path)
            question = vectorizer(tf.constant(questions[n])).numpy()

            if y is None:
                yield {"image": image, "question": question}
            else:
                yield {"image": image, "question": question}, int(y[n])

    output_signature_x = {
        "image": tf.TensorSpec(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
        "question": tf.TensorSpec(shape=(seq_len,), dtype=tf.int64),
    }

    if y is None:
        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature_x)
    else:
        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                output_signature_x,
                tf.TensorSpec(shape=(), dtype=tf.int64),
            ),
        )

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------

def build_model(num_classes, vocab_size=4000, seq_len=80):
    """
    Rede multimodal: imagem + pergunta -> resposta.
    """
    image_input = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="image")
    question_input = layers.Input(shape=(seq_len,), dtype=tf.int64, name="question")

    x_img = layers.Conv2D(32, 3, activation="relu", padding="same")(image_input)
    x_img = layers.MaxPooling2D()(x_img)
    x_img = layers.Conv2D(64, 3, activation="relu", padding="same")(x_img)
    x_img = layers.MaxPooling2D()(x_img)
    x_img = layers.Conv2D(128, 3, activation="relu", padding="same")(x_img)
    x_img = layers.MaxPooling2D()(x_img)
    x_img = layers.Conv2D(256, 3, activation="relu", padding="same")(x_img)
    x_img = layers.GlobalAveragePooling2D()(x_img)
    x_img = layers.Dense(256, activation="relu")(x_img)
    x_img = layers.Dropout(0.25)(x_img)

    x_txt = layers.Embedding(input_dim=vocab_size, output_dim=128, mask_zero=True)(question_input)
    x_txt = layers.Bidirectional(layers.GRU(128))(x_txt)
    x_txt = layers.Dense(128, activation="relu")(x_txt)
    x_txt = layers.Dropout(0.25)(x_txt)

    x = layers.Concatenate()([x_img, x_txt])
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    output = layers.Dense(num_classes, activation="softmax", name="answer")(x)

    model = Model(inputs={"image": image_input, "question": question_input}, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------------------
# Extracao de PDF
# ---------------------------------------------------------------------

def extract_pdf(pdf_path, images_dir, dpi=150):
    import fitz

    pdf_path = Path(pdf_path)
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    for i, page in enumerate(tqdm(doc, desc="Extraindo paginas do PDF")):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        pix.save(images_dir / f"{i:04d}.png")

    print(f"OK: {len(doc)} imagens salvas em {images_dir}")


# ---------------------------------------------------------------------
# Treino e predicao
# ---------------------------------------------------------------------

def command_download(args):
    project_root = get_project_root(args.project_root)
    print_environment_summary(project_root)

    if args.create_project_dirs:
        project_root.mkdir(parents=True, exist_ok=True)
        (project_root / "scripts").mkdir(exist_ok=True)
        (project_root / "src").mkdir(exist_ok=True)
        (project_root / "notebooks").mkdir(exist_ok=True)
        (project_root / "tests").mkdir(exist_ok=True)

    download_kaggle_dataset(
        project_root=project_root,
        kaggle_json=args.kaggle_json,
        force=args.force,
    )


def command_train(args):
    project_root = get_project_root(args.project_root)
    print_environment_summary(project_root)

    if args.download:
        download_kaggle_dataset(
            project_root=project_root,
            kaggle_json=args.kaggle_json,
            force=args.force_download,
        )

    validate_dataset_structure(project_root)

    default_paths = resolve_default_paths(project_root)

    jsonl = Path(args.jsonl).expanduser().resolve() if args.jsonl else default_paths["train_jsonl"]
    images_dir = Path(args.images_dir).expanduser().resolve() if args.images_dir else default_paths["train_images_dir"]
    model_dir = get_model_dir(project_root, args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Treinamento")
    print("-----------")
    print(f"JSONL de treino: {jsonl}")
    print(f"Diretorio de imagens de treino: {images_dir}")
    print(f"Diretorio do modelo: {model_dir}")
    print()

    df = read_jsonl(jsonl, require_answer=True)
    df["answer"] = df["answer"].astype(str)
    df["question"] = df["question"].map(normalize_question)

    print(f"Amostras: {len(df)}")
    print(f"Indices: {df['index'].min()} a {df['index'].max()}")
    print("Classes de resposta:", sorted(df["answer"].unique(), key=lambda x: (len(x), x)))

    stratify = df["answer"] if df["answer"].nunique() > 1 else None

    train_df, val_df = train_test_split(
        df,
        test_size=args.validation_split,
        random_state=args.seed,
        stratify=stratify,
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["answer"])

    vectorizer = build_text_vectorizer(
        train_df["question"].tolist(),
        max_tokens=args.max_tokens,
        seq_len=args.seq_len,
    )

    train_ds = make_dataset(
        train_df,
        images_dir,
        vectorizer,
        label_encoder,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_ds = make_dataset(
        val_df,
        images_dir,
        vectorizer,
        label_encoder,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = build_model(
        num_classes=len(label_encoder.classes_),
        vocab_size=args.max_tokens,
        seq_len=args.seq_len,
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_dir / "best_model.keras",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=args.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(model_dir / "final_model.keras")

    vectorizer_config = vectorizer.get_config()
    vectorizer_vocabulary = vectorizer.get_vocabulary()

    with (model_dir / "vectorizer.pkl").open("wb") as f:
        pickle.dump(
            {
                "config": vectorizer_config,
                "vocabulary": vectorizer_vocabulary,
            },
            f,
        )

    with (model_dir / "label_encoder.pkl").open("wb") as f:
        pickle.dump(label_encoder, f)

    hist = pd.DataFrame(history.history)
    hist.to_csv(model_dir / "training_history.csv", index=False)

    print()
    print(f"Modelo e metadados salvos em: {model_dir}")


def load_vectorizer(model_dir):
    with (Path(model_dir) / "vectorizer.pkl").open("rb") as f:
        data = pickle.load(f)

    vectorizer = layers.TextVectorization.from_config(data["config"])
    vectorizer.set_vocabulary(data["vocabulary"])

    return vectorizer


def command_predict(args):
    project_root = get_project_root(args.project_root)
    print_environment_summary(project_root)

    if args.download:
        download_kaggle_dataset(
            project_root=project_root,
            kaggle_json=args.kaggle_json,
            force=args.force_download,
        )

    validate_dataset_structure(project_root)

    default_paths = resolve_default_paths(project_root)

    jsonl = Path(args.jsonl).expanduser().resolve() if args.jsonl else default_paths["test_jsonl"]
    images_dir = Path(args.images_dir).expanduser().resolve() if args.images_dir else default_paths["test_images_dir"]
    model_dir = get_model_dir(project_root, args.model_dir)

    output = Path(args.output)
    if not output.is_absolute():
        output = project_root / output

    print("Predicao")
    print("--------")
    print(f"JSONL de teste: {jsonl}")
    print(f"Diretorio de imagens de teste: {images_dir}")
    print(f"Diretorio do modelo: {model_dir}")
    print(f"Saida: {output}")
    print()

    df = read_jsonl(jsonl, require_answer=False)
    df["question"] = df["question"].map(normalize_question)

    model_path = model_dir / "best_model.keras"
    if not model_path.exists():
        model_path = model_dir / "final_model.keras"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado em {model_dir}. "
            "Treine primeiro com o comando train."
        )

    model = tf.keras.models.load_model(model_path)
    vectorizer = load_vectorizer(model_dir)

    with (model_dir / "label_encoder.pkl").open("rb") as f:
        label_encoder = pickle.load(f)

    ds = make_dataset(
        df,
        images_dir,
        vectorizer,
        label_encoder=None,
        batch_size=args.batch_size,
        shuffle=False,
    )

    prob = model.predict(ds)
    pred_ids = prob.argmax(axis=1)
    pred_answers = label_encoder.inverse_transform(pred_ids)
    confidence = prob.max(axis=1)

    out = pd.DataFrame(
        {
            "index": df["index"],
            "question": df["question"],
            "predicted_answer": pred_answers,
            "confidence": confidence,
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    print(f"Predicoes salvas em: {output}")


def command_train_predict(args):
    command_train(args)

    predict_args = argparse.Namespace(**vars(args))
    predict_args.jsonl = None
    predict_args.images_dir = None
    predict_args.download = False
    predict_args.output = args.output
    command_predict(predict_args)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def add_common_project_args(parser):
    parser.add_argument(
        "--project-root",
        default=None,
        help=(
            "Raiz do projeto. "
            "Local: normalmente nao precisa informar. "
            "Colab: recomenda-se /content/PCS5022_2026."
        ),
    )
    parser.add_argument(
        "--kaggle-json",
        default=None,
        help="Caminho para o arquivo kaggle.json, se precisar configurar credenciais.",
    )


def add_download_flags(parser):
    parser.add_argument(
        "--download",
        action="store_true",
        help="Baixa o dataset do Kaggle antes de executar o comando.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Forca novo download do dataset mesmo se a estrutura ja existir.",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Treina e prediz VQA multimodal para circuitos de portas logicas "
            "em ambiente local ou Google Colab."
        )
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_download = sub.add_parser("download-data", help="Baixa o dataset do Kaggle.")
    add_common_project_args(p_download)
    p_download.add_argument("--force", action="store_true", help="Forca novo download.")
    p_download.add_argument(
        "--create-project-dirs",
        action="store_true",
        help="Cria diretorios basicos do projeto no Colab ou local.",
    )
    p_download.set_defaults(func=command_download)

    p_extract = sub.add_parser("extract-pdf", help="Extrai cada pagina do PDF como PNG.")
    p_extract.add_argument("--pdf", required=True)
    p_extract.add_argument("--images-dir", required=True)
    p_extract.add_argument("--dpi", type=int, default=150)
    p_extract.set_defaults(func=lambda args: extract_pdf(args.pdf, args.images_dir, args.dpi))

    p_train = sub.add_parser("train", help="Treina o modelo multimodal.")
    add_common_project_args(p_train)
    add_download_flags(p_train)
    p_train.add_argument("--jsonl", default=None, help="JSONL de treino. Se omitido, usa dataset/train_dataset/text/*.jsonl.")
    p_train.add_argument("--images-dir", default=None, help="Pasta de imagens de treino. Se omitida, usa dataset/train_dataset/images.")
    p_train.add_argument("--model-dir", default=DEFAULT_MODEL_DIR_NAME)
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--validation-split", type=float, default=0.2)
    p_train.add_argument("--max-tokens", type=int, default=4000)
    p_train.add_argument("--seq-len", type=int, default=80)
    p_train.add_argument("--patience", type=int, default=7)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.set_defaults(func=command_train)

    p_predict = sub.add_parser("predict", help="Gera predicoes para JSONL de teste.")
    add_common_project_args(p_predict)
    add_download_flags(p_predict)
    p_predict.add_argument("--jsonl", default=None, help="JSONL de teste. Se omitido, usa dataset/test_dataset/text/*.jsonl.")
    p_predict.add_argument("--images-dir", default=None, help="Pasta de imagens de teste. Se omitida, usa dataset/test_dataset/images.")
    p_predict.add_argument("--model-dir", default=DEFAULT_MODEL_DIR_NAME)
    p_predict.add_argument("--output", default="predictions.csv")
    p_predict.add_argument("--batch-size", type=int, default=32)
    p_predict.set_defaults(func=command_predict)

    p_all = sub.add_parser("train-predict", help="Baixa, treina e prediz em sequencia.")
    add_common_project_args(p_all)
    add_download_flags(p_all)
    p_all.add_argument("--jsonl", default=None, help="JSONL de treino. Se omitido, usa dataset/train_dataset/text/*.jsonl.")
    p_all.add_argument("--images-dir", default=None, help="Pasta de imagens de treino. Se omitida, usa dataset/train_dataset/images.")
    p_all.add_argument("--model-dir", default=DEFAULT_MODEL_DIR_NAME)
    p_all.add_argument("--output", default="predictions.csv")
    p_all.add_argument("--epochs", type=int, default=30)
    p_all.add_argument("--batch-size", type=int, default=32)
    p_all.add_argument("--validation-split", type=float, default=0.2)
    p_all.add_argument("--max-tokens", type=int, default=4000)
    p_all.add_argument("--seq-len", type=int, default=80)
    p_all.add_argument("--patience", type=int, default=7)
    p_all.add_argument("--seed", type=int, default=42)
    p_all.set_defaults(func=command_train_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
