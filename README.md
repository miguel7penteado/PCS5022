# Deep Learning kaggle challenge


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