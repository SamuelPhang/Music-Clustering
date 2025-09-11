#!/usr/bin/env python3
"""
Startup script for the Music Clustering web server.
"""

import os
import sys
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        # 'tensorflow',
        'fastapi',
        'uvicorn',
        'faiss',
        'lmdb',
        'essentia',
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.info("Please install them using: pip install -r requirements.txt")
        return False
    
    return True


def check_model_files():
    """Check if required model files exist."""
    required_files = [
        'audioset.index',
        'audioset.lmdb',
        'model/audioset-vggish-3.pb'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        logger.info("Please ensure you have:")
        logger.info("1. Built the FAISS index (audioset.index and audioset.lmdb)")
        logger.info("2. Downloaded the VGGish model (model/audioset-vggish-3.pb)")
        return False
    
    return True


def main():
    """Main startup function."""
    logger.info("🚀 Starting Music Clustering Web Server...")
    
    # Check dependencies
    logger.info("📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Check model files
    logger.info("🤖 Checking model files...")
    if not check_model_files():
        sys.exit(1)
    
    # Start the server
    logger.info("🌐 Starting web server...")
    logger.info("📖 API documentation will be available at: http://localhost:8000/docs")
    logger.info("🔍 Health check: http://localhost:8000/health")
    logger.info("🛑 Press Ctrl+C to stop the server")
    
    try:
        # Change to src directory and run the server
        os.chdir('src')
        subprocess.run([
            sys.executable, 'web_server.py'
        ], check=True)
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

