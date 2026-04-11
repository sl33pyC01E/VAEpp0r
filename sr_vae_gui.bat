@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python -m experiments.sr_vae_gui
