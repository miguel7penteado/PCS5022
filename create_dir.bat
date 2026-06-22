@echo off
setlocal

REM Nome da pasta principal do projeto
set "BASE_DIR=meu_projeto"

echo Criando estrutura do projeto: %BASE_DIR%
echo.

REM Criar pastas
mkdir "%BASE_DIR%"
mkdir "%BASE_DIR%\src"
mkdir "%BASE_DIR%\src\meu_pacote"
mkdir "%BASE_DIR%\notebooks"
mkdir "%BASE_DIR%\data"
mkdir "%BASE_DIR%\data\raw"
mkdir "%BASE_DIR%\data\processed"
mkdir "%BASE_DIR%\scripts"
mkdir "%BASE_DIR%\tests"

REM Criar arquivos vazios
type nul > "%BASE_DIR%\src\meu_pacote\__init__.py"
type nul > "%BASE_DIR%\src\meu_pacote\preprocessamento.py"
type nul > "%BASE_DIR%\src\meu_pacote\modelo.py"
type nul > "%BASE_DIR%\src\meu_pacote\treino.py"

type nul > "%BASE_DIR%\notebooks\01_exploracao.ipynb"
type nul > "%BASE_DIR%\notebooks\02_treinamento.ipynb"

type nul > "%BASE_DIR%\data\README.md"

type nul > "%BASE_DIR%\scripts\baixar_kaggle.py"
type nul > "%BASE_DIR%\scripts\publicar_kaggle.py"

type nul > "%BASE_DIR%\tests\test_modelo.py"

type nul > "%BASE_DIR%\pyproject.toml"
type nul > "%BASE_DIR%\README.md"
type nul > "%BASE_DIR%\LICENSE"
type nul > "%BASE_DIR%\CITATION.cff"
type nul > "%BASE_DIR%\.zenodo.json"
type nul > "%BASE_DIR%\dataset-metadata.json"
type nul > "%BASE_DIR%\.gitignore"

echo.
echo Estrutura criada com sucesso!
echo.

tree "%BASE_DIR%" /F

endlocal
pause