#!/bin/bash
# Build script for MEEG application

# Exit on error
set -e

echo "==================================="
echo "MEEG Application Build Script"
echo "==================================="

# Check if PyInstaller is installed
if ! command -v pyinstaller &> /dev/null; then
    echo "PyInstaller not found. Installing..."
    pip install pyinstaller
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/MEEG
rm -rf dist/MEEG
rm -rf dist/MEEG.app

# Build the application
echo "Building MEEG application..."
pyinstaller MEEG.spec --clean

echo ""
echo "==================================="
echo "Build complete!"
echo "==================================="
echo ""
echo "The application bundle is located at:"
echo "  dist/MEEG.app (macOS application)"
echo "  dist/MEEG/ (folder with executable)"
echo ""
echo "To run the app:"
echo "  open dist/MEEG.app"
echo ""
echo "To distribute, you can:"
echo "  1. Zip the .app bundle: zip -r MEEG.zip dist/MEEG.app"
echo "  2. Create a DMG (optional): hdiutil create -srcfolder dist/MEEG.app MEEG.dmg"
echo ""
