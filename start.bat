@echo off
echo ======================================================
echo    Starting TGI LLM Serving Dashboard
echo ======================================================

echo [1/2] Starting Backend (FastAPI on port 8000)...
start "TGI Backend (FastAPI)" cmd /k "cd /d "%~dp0backend" && venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000"

echo [2/2] Starting Frontend (HTTP Server on port 5500)...
start "TGI Frontend (Web)" cmd /k "cd /d "%~dp0frontend" && ..\backend\venv\Scripts\python.exe -m http.server 5500"

echo.
echo Servers are launching:
echo  - Backend API:  http://localhost:8000
echo  - Frontend UI:  http://localhost:5500
echo.
timeout /t 3 >nul
start http://localhost:5500
