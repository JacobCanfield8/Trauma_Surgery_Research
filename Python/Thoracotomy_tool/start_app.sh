#!/bin/bash

LOGFILE="start_app.log"

# Function to check the status of the last executed command
check_status() {
  if [ $? -ne 0 ]; then
    echo "Error encountered. Check the log file: $LOGFILE"
    exit 1
  fi
}

echo "Starting application..." > $LOGFILE

# Kill any process using port 8080
echo "Killing any process using port 8080..." | tee -a $LOGFILE
PIDS=$(lsof -t -i:8080)
if [ ! -z "$PIDS" ]; then
  kill -9 $PIDS >> $LOGFILE 2>&1
  check_status
fi

# Start Gunicorn
echo "Starting Gunicorn..." | tee -a $LOGFILE
gunicorn --bind 0.0.0.0:8080 wsgi:app >> $LOGFILE 2>&1 &
GUNICORN_PID=$!
sleep 5  # Give Gunicorn time to start
echo "Gunicorn started with PID $GUNICORN_PID" | tee -a $LOGFILE
check_status

# Reload nginx
echo "Reloading nginx..." | tee -a $LOGFILE
sudo nginx -s reload >> $LOGFILE 2>&1
check_status
echo "nginx reloaded successfully." | tee -a $LOGFILE

# Start ngrok
echo "Starting ngrok..." | tee -a $LOGFILE
ngrok http 8080 >> $LOGFILE 2>&1 &
check_status
echo "ngrok started successfully." | tee -a $LOGFILE

echo "All services started. Check $LOGFILE for details."

