#!/bin/bash
set -e

# Try apt-get first (Debian/Ubuntu)
if command -v apt-get &> /dev/null; then
    echo "Installing FFmpeg using apt-get..."
    apt-get update
    apt-get install -y ffmpeg
# Try yum next (RHEL/CentOS)
elif command -v yum &> /dev/null; then
    echo "Installing FFmpeg using yum..."
    yum install -y epel-release
    yum install -y ffmpeg
# Try apk next (Alpine)
elif command -v apk &> /dev/null; then
    echo "Installing FFmpeg using apk..."
    apk add --no-cache ffmpeg
else
    echo "No package manager found, trying to install FFmpeg from static build..."
    mkdir -p /tmp/ffmpeg
    cd /tmp/ffmpeg
    wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
    tar xf ffmpeg-release-amd64-static.tar.xz
    cd ffmpeg-*-static
    cp ffmpeg ffprobe /usr/local/bin/
    chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe
fi

# Verify installation
echo "FFmpeg installation completed, checking version:"
which ffmpeg || echo "FFmpeg command not found!"
ffmpeg -version || echo "FFmpeg not working correctly!"
