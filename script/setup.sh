#!/bin/bash
# =============================================================================
# VacuumBot Setup Script for Raspberry Pi
# Run as: sudo bash setup.sh
# =============================================================================
set -e

INSTALL_DIR="/home/pi/vacuumbot"
SERVICE_FILE="/etc/systemd/system/vacuumbot.service"
LOG_DIR="/var/log"
DATA_DIR="/var/lib/vacuumbot"

echo "============================================"
echo "  VacuumBot Install"
echo "============================================"

# System dependencies
apt-get update -q
apt-get install -y python3 python3-pip python3-venv libopenblas-dev

# Create install directory
mkdir -p "$INSTALL_DIR"
cp -r backend  "$INSTALL_DIR/"
cp -r frontend "$INSTALL_DIR/"
cp -r config   "$INSTALL_DIR/"
chown -R pi:pi "$INSTALL_DIR"

# Create data directory
mkdir -p "$DATA_DIR"
chown -R pi:pi "$DATA_DIR"

# Python virtual environment
sudo -u pi python3 -m venv "$INSTALL_DIR/venv"
sudo -u pi "$INSTALL_DIR/venv/bin/pip" install --upgrade pip -q
sudo -u pi "$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/backend/requirements.txt" -q

# Serial port permissions
usermod -aG dialout pi

# Update main.py to look in right place for frontend
sed -i "s|static_folder='../frontend'|static_folder='$INSTALL_DIR/frontend'|g" \
    "$INSTALL_DIR/backend/main.py"

# Systemd service
cp config/vacuumbot.service "$SERVICE_FILE"
systemctl daemon-reload
systemctl enable vacuumbot
systemctl restart vacuumbot

echo ""
echo "============================================"
echo "  Installation complete!"
echo ""
echo "  Web UI:   http://$(hostname -I | awk '{print $1}'):5000"
echo "  Logs:     journalctl -fu vacuumbot"
echo "  Status:   systemctl status vacuumbot"
echo "============================================"
