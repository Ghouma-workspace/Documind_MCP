@echo off
REM DocuMind Project Runner
REM Convenient script to run various project commands

SETLOCAL EnableDelayedExpansion

:menu
cls
echo ========================================
echo     DocuMind - Document Automation
echo ========================================
echo.
echo 1. Run Demo
echo 2. Start API Server
echo 3. Start Web UI
echo 4. Start Both (API + UI)
echo 5. Run Tests
echo 6. Check System Health
echo 7. View Logs
echo 8. Clear Data
echo 9. Docker - Build and Run
echo 0. Exit
echo.
set /p choice="Enter your choice (0-9): "

if "%choice%"=="1" goto demo
if "%choice%"=="2" goto api
if "%choice%"=="3" goto ui
if "%choice%"=="4" goto both
if "%choice%"=="5" goto tests
if "%choice%"=="6" goto health
if "%choice%"=="7" goto logs
if "%choice%"=="8" goto clear
if "%choice%"=="9" goto docker
if "%choice%"=="0" goto end
goto menu

:demo
echo.
echo Running demo...
echo.
call .venv\Scripts\activate.bat
python demo.py
echo.
pause
goto menu

:api
echo.
echo Starting API server on http://localhost:8000
echo Press Ctrl+C to stop
echo.
call .venv\Scripts\activate.bat
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
goto menu

:ui
echo.
echo Starting Web UI on http://localhost:8501
echo Press Ctrl+C to stop
echo.
call .venv\Scripts\activate.bat
streamlit run app/ui.py
goto menu

:both
echo.
echo Starting API and UI...
echo.
echo Opening API in new window...
start cmd /k "call .venv\Scripts\activate.bat && uvicorn app.main:app --reload"
timeout /t 5
echo Opening UI in new window...
start cmd /k "call .venv\Scripts\activate.bat && streamlit run app/ui.py"
echo.
echo Both services started in separate windows!
pause
goto menu

:tests
echo.
echo Running tests...
echo.
call .venv\Scripts\activate.bat
pytest tests/ -v
echo.
pause
goto menu

:health
echo.
echo Checking system health...
echo.
curl -s http://localhost:8000/health | python -m json.tool
if errorlevel 1 (
    echo.
    echo ERROR: API server is not running!
    echo Please start the API server first (option 2)
)
echo.
pause
goto menu

:logs
echo.
echo Viewing recent logs...
echo.
if exist documind.log (
    powershell -command "Get-Content documind.log -Tail 50"
) else (
    echo No log file found. Run the application first.
)
echo.
pause
goto menu

:clear
echo.
echo WARNING: This will delete all indexed documents and outputs!
set /p confirm="Are you sure? (yes/no): "
if /i "%confirm%"=="yes" (
    echo Clearing data...
    if exist data\faiss_index rmdir /s /q data\faiss_index
    if exist data\document_store rmdir /s /q data\document_store
    del /q data\outputs\* 2>nul
    echo Data cleared!
) else (
    echo Cancelled.
)
echo.
pause
goto menu

:docker
echo.
echo Building and running Docker containers...
echo.
docker-compose build
docker-compose up -d
echo.
echo Containers started!
echo API: http://localhost:8000
echo UI: http://localhost:8501
echo.
echo To stop: docker-compose down
echo.
pause
goto menu

:end
echo.
echo Goodbye!
exit /b 0
