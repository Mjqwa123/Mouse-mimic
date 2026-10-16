@echo off
rem Builds MouseMimic.exe (single file) into dist\ and copies it to the Desktop.
cd /d "%~dp0"
python -m PyInstaller --noconfirm --clean --onefile --windowed --name MouseMimic ^
  --icon icon.ico --add-data "icon.ico;." --splash icon.png ^
  --exclude-module matplotlib --exclude-module IPython --exclude-module pandas ^
  app.py || exit /b 1
copy /y dist\MouseMimic.exe "%USERPROFILE%\OneDrive\Desktop\MouseMimic.exe"
