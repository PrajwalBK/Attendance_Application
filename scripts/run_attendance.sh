#!/bin/bash

# Attendance System Launcher for Raspberry Pi
# -----------------------------------------

# Use absolute path to guarantee we are in the right place
APP_DIR="/home/prosper123/desktop_application-v1"
cd "$APP_DIR" || exit 1

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Export Python Path
export PYTHONPATH=$APP_DIR

# Set the log file path (to the desktop so it's easy to find)
LOG_FILE="/home/prosper123/Desktop/pi_gui_error.log"

# Run the GUI and log output to BOTH the terminal and the log file
echo "Starting Face Attendance GUI (this may take a minute on first run)..."
echo "--- Application Start: $(date) ---" > "$LOG_FILE"
python3 ui/modern_gui.py 2>&1 | tee -a "$LOG_FILE"

# Keep terminal open if it crashes quickly so user can read it
if [ $? -ne 0 ]; then
    echo ""
    echo "====================================="
    echo "CRASH DETECTED!"
    echo "Please read the log file that was saved to your Desktop:"
    echo "    $LOG_FILE"
    echo "====================================="
    read -p "Press Enter to close this window..."
fi
