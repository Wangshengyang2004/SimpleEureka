import sys
from loguru import logger

import os
import shutil
def check_system_encoding():
    """Check the system encoding and return False if not UTF-8."""
    if sys.getdefaultencoding() != 'utf-8':
        return False

def clean_folder(path):
    """Remove all files in a folder."""
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.info(e)
    logger.info(f"Folder cleaned: {path}")

def copy_folder(src, dest):
    """Copy all files from src to dest."""
    try:
        shutil.copytree(src, dest)
        logger.info(f"Folder copied: {src} -> {dest}")
    except Exception as e:
        logger.info(e)
    
def copy_folder_sub(src, dest):
    """Copy all files under src to dest."""
    for file in os.listdir(src):
        file_path = os.path.join(src, file)
        try:
            if os.path.isfile(file_path):
                shutil.copy(file_path, dest)
        except Exception as e:
            logger.info(e)
    logger.info(f"Folder copied: {src} -> {dest}")