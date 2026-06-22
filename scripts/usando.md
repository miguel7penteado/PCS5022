# Como baixar e enviar imagens ao kaggle


### Download dataset

```
python baixar_kaggle.py
```

### Upload dataset

```
python preparar_kaggle_dataset.py
kaggle datasets create -p .
```


### Atualizar dataset

```
kaggle datasets version -p . -m "Atualização do dataset"
```
