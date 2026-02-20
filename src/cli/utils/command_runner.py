import os
import shlex
import subprocess
import threading
from pathlib import Path
from typing import Optional, Tuple

from src.utils.logging import get_logger

logger = get_logger(__name__)

class CommandRunner:
    """Centralized command execution utility"""
    
    @staticmethod
    def run_simple(command_str: str, cwd: Path = None) -> Tuple[str, str, int]:
        """Simple command execution - no streaming"""
        posix = os.name == 'posix'
        command_list = shlex.split(command_str, posix=posix)
        logger.debug(f"Executing command: {command_str}")
        
        process = subprocess.Popen(
            command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd
        )
        
        stdout, stderr = process.communicate()
        exit_code = process.returncode
        
        logger.debug(f"Command completed with exit code: {exit_code}")
        return stdout, stderr, exit_code
    
    @staticmethod
    def run_streaming(command_str: str, cwd: Path = None) -> Tuple[str, str, int]:
        """Run command with real-time output streaming and proper exit code handling"""
        posix = os.name == 'posix'
        command_list = shlex.split(command_str, posix=posix)
        
        logger.debug(f"Executing command: {command_str}")
        
        process = subprocess.Popen(
            command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
            cwd=cwd
        )

        stdout_lines = []
        stderr_lines = []

        def _read_stream(stream, collector, prefix=""):
            """Read from stream and collect lines while printing in real-time"""
            try:
                for line in iter(stream.readline, ''):
                    if line:
                        collector.append(line)
                        # Print with prefix for clarity
                        logger.debug(f"{prefix}{line.rstrip()}")
                        for handler in logger.handlers:
                            handler.flush()
            except Exception as e:
                logger.debug(f"Error reading stream: {e}")
            finally:
                stream.close()

        # Start threads for non-blocking reads
        stdout_thread = threading.Thread(
            target=_read_stream, 
            args=(process.stdout, stdout_lines, "")
        )
        stderr_thread = threading.Thread(
            target=_read_stream, 
            args=(process.stderr, stderr_lines, "")
        )
        
        stdout_thread.start()
        stderr_thread.start()

        # Wait for command to finish
        try:
            exit_code = process.wait()
            stdout_thread.join(timeout=5.0)  # Add timeout to prevent hanging
            stderr_thread.join(timeout=5.0)
            
        except KeyboardInterrupt:
            logger.info("Command interrupted by user")
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            
            stdout_thread.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)
            raise
        
        stdout_text = ''.join(stdout_lines)
        stderr_text = ''.join(stderr_lines)
        
        logger.debug(f"Command completed with exit code: {exit_code}")
        
        return stdout_text, stderr_text, exit_code