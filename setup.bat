@echo off
echo Setting up P9 Few-Shot Labeler Environment...

:: Check if venv exists, create if not
if not exist "venv\Scripts\activate.bat" (
    echo Creating Python virtual environment...
    python -m venv venv
)

:: Activate and install dependencies
call venv\Scripts\activate.bat
echo Installing dependencies from requirements.txt...
pip install --upgrade pip
pip install -r requirements.txt

:: Create necessary directories
mkdir data\raw 2>nul
mkdir data\gold_standard 2>nul
mkdir data\processed 2>nul
mkdir models\distilbert_finetuned 2>nul

echo Environment ready. Launching Streamlit app...
streamlit run app.py
pause
