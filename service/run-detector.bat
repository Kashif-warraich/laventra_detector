@echo off
REM Launches the Laventra detector. Used by the Task Scheduler task so it
REM survives reboots and runs headless. Paths are relative to this .bat
REM (service\), so it works regardless of where the repo lives.

cd /d "%~dp0.."

REM If the exported Windows CA bundle exists, trust it so first-run model
REM downloads (Hugging Face / EasyOCR) succeed behind Avast TLS inspection.
set "CA=%~dp0..\win-ca-bundle.pem"
if exist "%CA%" (
  set "SSL_CERT_FILE=%CA%"
  set "REQUESTS_CA_BUNDLE=%CA%"
  set "CURL_CA_BUNDLE=%CA%"
)

"%~dp0..\venv\Scripts\python.exe" -u main.py
