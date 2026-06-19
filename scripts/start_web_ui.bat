@echo off
echo ==============================================
echo STARTING VISION ATTENDANCE AI (WEB UI)
echo ==============================================

echo [1] Starting Local Camera Backend (FastAPI)...
start "Attendance AI Backend" cmd /k "venv\Scripts\activate.bat && python api.py"

echo [2] Starting Frontend Service (React/Vite)...
cd frontend
start "Attendance AI Frontend" cmd /k "npm run dev"

echo Done! The UI should open in your browser shortly.
pause
