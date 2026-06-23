# Deep Learning kaggle challenge

## The Neural Network

```cmd
python scripts\miguel_neural_network.py download-data
python scripts\miguel_neural_network.py train
python scripts\miguel_neural_network.py predict
python scripts\miguel_neural_network.py train-predict --download
```

### Local Computer

- Download files (dataset)

```cmd
python scripts\miguel_neural_network.py download-data
```


- train model with dataset

```cmd
python scripts\miguel_neural_network.py train
```

- predict data (test model)

```cmd
python scripts\miguel_neural_network.py predict
```

- Do all at once 

```cmd
python scripts\miguel_neural_network.py train-predict --download
```

### Remote Computer (Google Colab)

- First put yout `kaggle.json` from your kaggle inside your google colab

```python
!python /content/PCS5022_2026/scripts/miguel_neural_network.py train-predict \
    --project-root /content/PCS5022_2026 \
    --download \
    --kaggle-json /content/kaggle.json \
    --epochs 10
```

The dataset will be saved on `/content/PCS5022_2026/dataset/`

The model wil be created on `/content/PCS5022_2026/model_logic_vqa/`

Notes: 

- if you need to download dataset from kaggle again do that
```python
python scripts\logic_vqa_portable.py download-data --force
```

- train model with less epochs
```python
python scripts\logic_vqa_portable.py train --epochs 5
```

- Generate predictions on another file:
```python
python scripts\logic_vqa_portable.py predict --output resultados_teste.csv
```



## The Dataset

```
├───dataset
│   ├───test_dataset
│   │   ├───images
│   │   └───text
│   └───train_dataset
│       ├───images
│       └───text
├───kaggle_upload
│   ├───test_dataset
│   │   ├───images
│   │   └───text
│   └───train_dataset
│       ├───images
│       └───text
├───notebooks
├───scripts
├───src
│   └───meu_pacote
└───tests
```


First you need to download from your account kaggle´s API `%userprofile%\.kaggle\kaggle.json` on your computer.

Generating files on `kaggle_upload` dir before upload process

```cmd
python scripts\preparar_kaggle_dataset.py
```

Upload files to kaggle

```cmd
cd kaggle_upload
kaggle datasets create -p . --dir-mode zip
```

updateting files on kaggle

```cmd
cd kaggle_upload
kaggle datasets version -p . --dir-mode zip -m "Sending pictures and JSONL files updates"
```