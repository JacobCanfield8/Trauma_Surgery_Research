#!/bin/bash

LOGFILE="start_app.log"
LOGDIR="$(dirname $LOGFILE)"
mkdir -p $LOGDIR

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

# Prompt for sudo password at the start
echo "Please enter your sudo password to reload nginx:"
sudo -v

# Check if Nginx is running and reload or restart as necessary
if pgrep -x "nginx" > /dev/null
then
    echo "Reloading nginx..." | tee -a $LOGFILE
    sudo nginx -s reload >> $LOGFILE 2>&1
else
    echo "Nginx not running. Starting nginx..." | tee -a $LOGFILE
    sudo brew services start nginx >> $LOGFILE 2>&1
fi
check_status
echo "nginx reloaded or started successfully." | tee -a $LOGFILE

# Start ngrok using configuration file
echo "Starting ngrok..." | tee -a $LOGFILE
ngrok start --all >> $LOGFILE 2>&1 &
check_status
echo "ngrok started successfully." | tee -a $LOGFILE

echo "All services started. Check $LOGFILE for details."

