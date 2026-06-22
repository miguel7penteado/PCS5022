import subprocess
from pathlib import Path


DATASET = "miguelpenteado/meu-dataset-imagens-perguntas"
DEST = Path("data") / "dataset"


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

    subprocess.run(comando, check=True)

    print(f"Dataset baixado em: {DEST}")


if __name__ == "__main__":
    main()