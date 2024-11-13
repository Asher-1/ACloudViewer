import os
import subprocess
import logging
from datetime import datetime
import re
import sys
import threading
import queue


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

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def determine_log_level(line):
    if re.search(r'error|exception|fail', line, re.IGNORECASE):
        return logging.ERROR
    elif re.search(r'warning', line, re.IGNORECASE):
        return logging.WARNING
    else:
        return logging.INFO


def reader(pipe, queue):
    try:
        with pipe:
            for line in iter(pipe.readline, b''):
                queue.put(line)
    finally:
        queue.put(None)


def invoke_build_script(build_shell, python_version=None):
    if os.path.exists(build_shell):
        try:
            logging.info("Starting build process...")
            script_args = f'"{python_version}"' if python_version else ""
            cmd = f'powershell.exe -NoProfile -NonInteractive -File "{build_shell}" {script_args} {ACLOUDVIEWER_INSTALL}'

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

            q = queue.Queue()
            stdout_thread = threading.Thread(target=reader, args=[process.stdout, q])
            stderr_thread = threading.Thread(target=reader, args=[process.stderr, q])
            stdout_thread.start()
            stderr_thread.start()

            for line in iter(q.get, None):
                line = line.strip()
                if line:
                    log_level = determine_log_level(line)
                    logging.log(log_level, line)

            stdout_thread.join()
            stderr_thread.join()
            process.wait()
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


def build_gui_app():
    package_exists = any(file.startswith("ACloudViewer") and file.endswith(".exe")
                         for file in os.listdir(ACLOUDVIEWER_INSTALL))
    if not package_exists:
        logging.info("Start building ACloudViewer app...")
        if subprocess.call(
                ["powershell", "-File", REMOVE_FOLDERS_SHELL, "-FolderPath", CLOUDVIEWER_BUILD_DIR, "-y"]) == 0:
            logging.info("Build directory cleaned successfully")
        else:
            logging.error("Failed to clean build directory and please manually handle it...")
            return False

        result = invoke_build_script(WIN_APP_BUILD_SHELL)
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
        if subprocess.call(
                ["powershell", "-File", REMOVE_FOLDERS_SHELL, "-FolderPath", CLOUDVIEWER_BUILD_DIR, "-y"]) == 0:
            logging.info("Build directory cleaned successfully")
        else:
            logging.error("Failed to clean build directory and please manually handle it...")
            return False

        result = invoke_build_script(WIN_WHL_BUILD_SHELL, python_version)
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
    success = build_gui_app()
    if success:
        logging.info(f"Building package installed to {ACLOUDVIEWER_INSTALL}")
    else:
        exit(1)

    logging.info("\nStart to build wheel for python3.8-3.12 On Windows...\n")
    for version in ["3.8", "3.9", "3.10", "3.11", "3.12"]:
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

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    CLOUDVIEWER_BUILD_DIR = os.path.join(CLOUDVIEWER_SOURCE_ROOT, "build")
    logging.info(f"CLOUDVIEWER_BUILD_DIR: {CLOUDVIEWER_BUILD_DIR}")
    ACLOUDVIEWER_INSTALL = os.path.join("C:\\dev", "cloudViewer_install")
    logging.info(f"ACloudViewer_INSTALL PATH: {ACLOUDVIEWER_INSTALL}")

    WIN_APP_BUILD_SHELL = os.path.join(CLOUDVIEWER_SOURCE_ROOT, "scripts", "build_win_app.ps1")
    WIN_WHL_BUILD_SHELL = os.path.join(CLOUDVIEWER_SOURCE_ROOT, "scripts", "build_win_wheel.ps1")
    REMOVE_FOLDERS_SHELL = os.path.join(CLOUDVIEWER_SOURCE_ROOT, "scripts", "platforms", "windows",
                                        "remove_folders.ps1")

    build()
