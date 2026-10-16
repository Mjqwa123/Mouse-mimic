@echo off
rem Builds MouseMimic.exe (single file) into dist\, then copies it to your Desktop.
cd /d "%~dp0"
python -m PyInstaller --noconfirm --clean --onefile --windowed --name MouseMimic ^
  --icon icon.ico --add-data "icon.ico;." --splash icon.png ^
  --exclude-module matplotlib --exclude-module IPython --exclude-module pandas ^
  app.py || exit /b 1

echo.
echo Build succeeded: %~dp0dist\MouseMimic.exe

rem The copy is just a convenience, so a failure here doesn't fail the build.
rem Ask Windows where the Desktop really is: it may or may not be redirected to OneDrive.
set "DESKTOP="
for /f "usebackq delims=" %%D in (`powershell -NoProfile -Command "[Environment]::GetFolderPath('Desktop')"`) do set "DESKTOP=%%D"
if not defined DESKTOP goto nocopy
copy /y dist\MouseMimic.exe "%DESKTOP%\MouseMimic.exe" >nul 2>&1
if errorlevel 1 goto nocopy
echo Copied to:       %DESKTOP%\MouseMimic.exe
exit /b 0

:nocopy
echo Couldn't copy it to your Desktop; use the one in dist\ instead.
exit /b 0
