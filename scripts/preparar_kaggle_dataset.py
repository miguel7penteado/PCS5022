from pathlib import Path
import shutil
import json


# Raiz do projeto:
# Este arquivo está em scripts/preparar_kaggle_dataset.py
# Portanto, parents[1] aponta para a pasta principal do projeto.
ROOT = Path(__file__).resolve().parents[1]

# Nova estrutura:
# projeto/
# ├── dataset/
# │   ├── train_dataset/
# │   │   ├── images/
# │   │   └── text/
# │   └── test_dataset/
# │       ├── images/
# │       └── text/
# ├── scripts/
# ├── src/
# ├── notebooks/
# └── tests/
SOURCE = ROOT / "dataset"

# Pasta temporária que será enviada ao Kaggle
DEST = ROOT / "kaggle_upload"

# Altere estes campos conforme o nome real do seu dataset no Kaggle
DATASET_ID = "miguelpenteado/meu-dataset-imagens-perguntas"
DATASET_TITLE = "Meu Dataset de Imagens e Perguntas"
LICENSE_NAME = "CC-BY-SA-4.0"


def reset_dest():
    """
    Remove a pasta kaggle_upload antiga, se existir,
    e cria uma nova pasta limpa.
    """
    if DEST.exists():
        shutil.rmtree(DEST)

    DEST.mkdir(parents=True, exist_ok=True)


def validate_source():
    """
    Verifica se a estrutura esperada do dataset existe.
    """
    required_dirs = [
        SOURCE,
        SOURCE / "train_dataset",
        SOURCE / "train_dataset" / "images",
        SOURCE / "train_dataset" / "text",
        SOURCE / "test_dataset",
        SOURCE / "test_dataset" / "images",
        SOURCE / "test_dataset" / "text",
    ]

    for path in required_dirs:
        if not path.exists():
            raise FileNotFoundError(f"Pasta não encontrada: {path}")

    train_images = list((SOURCE / "train_dataset" / "images").glob("*.png"))
    test_images = list((SOURCE / "test_dataset" / "images").glob("*.png"))

    if not train_images:
        print("Aviso: nenhuma imagem PNG encontrada em train_dataset/images")

    if not test_images:
        print("Aviso: nenhuma imagem PNG encontrada em test_dataset/images")

    train_jsonl = list((SOURCE / "train_dataset" / "text").glob("*.jsonl"))
    test_jsonl = list((SOURCE / "test_dataset" / "text").glob("*.jsonl"))

    if not train_jsonl:
        print("Aviso: nenhum arquivo JSONL encontrado em train_dataset/text")

    if not test_jsonl:
        print("Aviso: nenhum arquivo JSONL encontrado em test_dataset/text")


def copy_dataset():
    """
    Copia train_dataset e test_dataset para kaggle_upload.
    """
    shutil.copytree(
        SOURCE / "train_dataset",
        DEST / "train_dataset"
    )

    shutil.copytree(
        SOURCE / "test_dataset",
        DEST / "test_dataset"
    )


def write_metadata():
    """
    Cria o arquivo dataset-metadata.json exigido pelo Kaggle.
    """
    metadata = {
        "title": DATASET_TITLE,
        "id": DATASET_ID,
        "licenses": [
            {
                "name": LICENSE_NAME
            }
        ]
    }

    metadata_path = DEST / "dataset-metadata.json"

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def write_readme():
    """
    Cria um README.md para acompanhar o dataset no Kaggle.
    """
    readme = f"""# {DATASET_TITLE}

Dataset contendo imagens PNG e arquivos JSONL com perguntas associadas às imagens.

## Estrutura do dataset

```text
train_dataset/
├── images/
└── text/

test_dataset/
├── images/
└── text/