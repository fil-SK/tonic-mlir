@echo off
echo ==============================
echo TONIC - Environment Setup
echo ==============================
echo.

echo Creating a virtual environment.
python -m venv .venv
echo Virtual environment created!
echo.

echo Installing dependencies from requirements.txt.
.venv\Scripts\python -m pip install -r requirements.txt
echo Necessary dependencies installed!
echo.

echo ==============================
echo Setup complete!
echo To activate the venv, run: .venv\Scripts\Activate.ps1
echo ==============================