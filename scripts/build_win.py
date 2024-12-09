# coding: utf-8
# Windows console: Set-ItemProperty HKCU:\Console VirtualTerminalLevel -Type DWORD 1

# Requirement:
# 1. Visual Studio 2019 community
# 2. vscode
# 3. Anaconda3
# 4. cuda11.8

import os
import subprocess
import logging
from datetime import datetime
import time
import re
import sys
import threading
import queue
import psutil


class ColorCodes:
    GREY = "\033[0;37m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    RED = "\033[0;31m"
    BOLD_RED = "\033[1;31m"
    RESET = "\033[0m"

class ColoredFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: ColorCodes.GREY + "%(asctime)s - %(levelname)s - %(message)s" + ColorCodes.RESET,
        logging.INFO: ColorCodes.GREEN + "%(asctime)s - %(levelname)s - %(message)s" + ColorCodes.RESET,
        logging.WARNING: ColorCodes.YELLOW + "%(asctime)s - %(levelname)s - %(message)s" + ColorCodes.RESET,
        logging.ERROR: ColorCodes.RED + "%(asctime)s - %(levelname)s - %(message)s" + ColorCodes.RESET,
        logging.CRITICAL: ColorCodes.BOLD_RED + "%(asctime)s - %(levelname)s - %(message)s" + ColorCodes.RESET
    }
    
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        msg = formatter.format(record)
        return self.ansi_escape.sub('', msg)

def determine_log_level(line):
    if re.search(r'\b(error|exception|fail(ed|ure)?)\b', line, re.IGNORECASE):
        if re.search(r'error_code\s*=\s*0', line, re.IGNORECASE):
            return logging.INFO
        if re.search(r'no\s+error', line, re.IGNORECASE):
            return logging.INFO
        return logging.ERROR
        
    if re.search(r'\bwarn(ing)?\b', line, re.IGNORECASE):
        return logging.WARNING
        
    return logging.INFO

def try_decode(byte_string):
    encodings = ['utf-8', 'gbk', 'cp1252']
    for encoding in encodings:
        try:
            return byte_string.decode(encoding)
        except UnicodeDecodeError:
            continue
    return byte_string.decode('utf-8', errors='replace')

def reader_wrapper(pipe, queue, stop_event):
    try:
        while not stop_event.is_set():
            line = pipe.readline()
            if not line:
                break
            decoded_line = try_decode(line)
            queue.put(decoded_line)
    finally:
        pipe.close()

def process_output(process, q):
    """
    Process the output from the queue and check for completion signal.
    """
    while True:
        try:
            line = q.get(timeout=1)
            if line:
                line = line.strip()
                log_level = determine_log_level(line)
                logging.log(log_level, line)
                if "BUILD_COMPLETE" in line:  # Check for completion signal
                    return True
        except queue.Empty:
            if process.poll() is not None:  # Check if process has ended
                return False
        except Exception as e:
            logging.log(logging.ERROR, str(e))
            return False

def invoke_build_script(build_shell, python_version, with_ml=False):
    if os.path.exists(build_shell):
        try:
            logging.info("Starting build process...")
            script_args = f'" {CloudViewerMLRoot}"' if with_ml else ""
            cmd = f'powershell.exe -NoProfile -NonInteractive -File "{build_shell}" {python_version} {ACLOUDVIEWER_INSTALL} {script_args}'

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False, bufsize=1)
            # process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, encoding='utf-8', errors='ignore')

            q = queue.Queue()
            stop_threads = threading.Event()

            stdout_thread = threading.Thread(target=reader_wrapper, args=[process.stdout, q, stop_threads])
            stderr_thread = threading.Thread(target=reader_wrapper, args=[process.stderr, q, stop_threads])
            
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()

            # Main loop to monitor the process
            while True:
                if process_output(process, q):
                    break
                if not psutil.pid_exists(process.pid):
                    break
                time.sleep(5)
        
            # Signal threads to stop
            stop_threads.set()
            
            # Wait for threads to finish
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                logging.warning("Process did not terminate in time, forcibly terminating...")
                process.kill()
            
            # Check the return code
            return_code = process.returncode
            success = return_code == 0
            
            if success:
                logging.info("Build succeeded")
            else:
                logging.error(f"Build failed with return code: {return_code}")
            return success
        except Exception as e:
            logging.error(f"Failed to execute build script: {str(e)}")
            return False
    else:
        logging.error(f"Error: Build script not found: {build_shell}")
        return False

def build_gui_app(python_version):
    package_exists = any(file.startswith("ACloudViewer") and file.endswith(".exe")
                         for file in os.listdir(ACLOUDVIEWER_INSTALL))
    if not package_exists:
        logging.info("Start building ACloudViewer app...")
        if (os.path.exists(CLOUDVIEWER_BUILD_DIR)):
            if subprocess.call(
                    ["powershell", "-File", REMOVE_FOLDERS_SHELL, "-FolderPath", CLOUDVIEWER_BUILD_DIR, "-y"]) == 0:
                logging.info("Build directory cleaned successfully")
            else:
                logging.error("Failed to clean build directory and please manually handle it...")
                return False

        result = invoke_build_script(WIN_APP_BUILD_SHELL, python_version, with_ml=False)
        if result:
            logging.info("Build succeeded")
            return True
        else:
            logging.error("Build failed")
            return False
    else:
        logging.info("Ignore ACloudViewer GUI app building due to have built before...")
        return True

def build_python_wheel(python_version):
    cp_version = f"cp{python_version.replace('.', '')}"
    pattern = re.compile(f"cloudViewer-.*-{cp_version}-{cp_version}-win_amd64\.whl$")
    wheel_exists = any(pattern.match(file) for file in os.listdir(ACLOUDVIEWER_INSTALL))
    if not wheel_exists:
        logging.info(f"Start building cloudViewer wheel for python{python_version}...")
        if (os.path.exists(CLOUDVIEWER_BUILD_DIR)):
            if subprocess.call(
                    ["powershell", "-File", REMOVE_FOLDERS_SHELL, "-FolderPath", CLOUDVIEWER_BUILD_DIR, "-y"]) == 0:
                logging.info("Build directory cleaned successfully")
            else:
                logging.error("Failed to clean build directory and please manually handle it...")
                return False
            
        if not os.path.exists(CloudViewerMLRoot):
            logging.error(f"Specified CloudViewerMLRoot does not exist: {CloudViewerMLRoot}")
            return False
        result = invoke_build_script(WIN_WHL_BUILD_SHELL, python_version, with_ml=True)
        if result:
            logging.info("Build succeeded")
            return True
        else:
            logging.error("Build failed")
            return False
    else:
        logging.info(f"Ignore cloudViewer wheel for python{python_version}...")
    return True

def build():
    for script in [WIN_APP_BUILD_SHELL, WIN_WHL_BUILD_SHELL, REMOVE_FOLDERS_SHELL]:
        if not os.path.exists(script):
            logging.error(f"Specified shell path does not exist: {script}")
            exit(1)

    logging.info("\nStart to build ACloudViewer GUI On Windows...\n")
    success = build_gui_app("3.11")
    if success:
        logging.info(f"Building package installed to {ACLOUDVIEWER_INSTALL}")
    else:
        exit(1)

    logging.info("\nStart to build wheel for python3.8-3.12 On Windows...\n")
    for version in ["3.8", "3.9", "3.10", "3.11", "3.12"]:
        logging.info("#" * 80)
        success = build_python_wheel(version)
        if success:
            logging.info(f"Successfully building cloudViewer on python{version}")
        else:
            exit(1)

    logging.info(f"All installed to {ACLOUDVIEWER_INSTALL}")


if __name__ == "__main__":
    CLOUDVIEWER_SOURCE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(CLOUDVIEWER_SOURCE_ROOT, "build.log")
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.info(f"Logging PATH: {log_path}")
    formatter = ColoredFormatter()
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    CLOUDVIEWER_BUILD_DIR = os.path.join(CLOUDVIEWER_SOURCE_ROOT, "build")
    logging.info(f"CLOUDVIEWER_BUILD_DIR: {CLOUDVIEWER_BUILD_DIR}")
    ACLOUDVIEWER_INSTALL = os.path.join("C:/dev", "cloudViewer_install")
    logging.info(f"ACloudViewer_INSTALL PATH: {ACLOUDVIEWER_INSTALL}")
    CloudViewerMLRoot = "C:/Users/asher/develop/code/CloudViewer/CloudViewer-ML"
    logging.info(f"CloudViewerMLRoot PATH: {CloudViewerMLRoot}")

    WIN_APP_BUILD_SHELL = os.path.join(CLOUDVIEWER_SOURCE_ROOT, "scripts", "build_win_app.ps1")
    WIN_WHL_BUILD_SHELL = os.path.join(CLOUDVIEWER_SOURCE_ROOT, "scripts", "build_win_wheel.ps1")
    REMOVE_FOLDERS_SHELL = os.path.join(CLOUDVIEWER_SOURCE_ROOT, "scripts", "platforms", "windows",
                                        "remove_folders.ps1")

    build()
