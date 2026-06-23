import subprocess
from pathlib import Path


KAGGLE_USERNAME = "miguel7penteado"
KAGGLE_DATASET_SLUG = "meu-dataset-imagens-perguntas"

DATASET = f"{KAGGLE_USERNAME}/{KAGGLE_DATASET_SLUG}"

# O dataset será baixado para a pasta dataset/
DEST = Path("dataset")


def main():
    DEST.mkdir(parents=True, exist_ok=True)

    comando = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET,
        "-p",
        str(DEST),
        "--unzip",
    ]

    print(f"Baixando dataset Kaggle: {DATASET}")
    print(f"Destino local: {DEST}")
    print()

    subprocess.run(comando, check=True)

    print()
    print(f"Dataset baixado com sucesso em: {DEST}")


if __name__ == "__main__":
    main()