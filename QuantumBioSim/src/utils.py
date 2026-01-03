import logging
import os
import sys

def setup_logging(log_file='evolution.log'):
    # Create a custom filter to avoid repetitive logs
    class NoRepetitiveLogs(logging.Filter):
        def __init__(self, max_messages=10000):
            super().__init__()
            self.logged_messages = set()
            self.max_messages = max_messages

        def filter(self, record):
            message = record.getMessage()
            if message in self.logged_messages:
                return False
            self.logged_messages.add(message)
            if len(self.logged_messages) > self.max_messages:
                self.logged_messages.pop()
            return True

    logger = logging.getLogger('quantum_bio_system')
    logger.setLevel(logging.INFO)
    
    # clear handlers if they exist
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler - force stdout to avoid PowerShell stderr quirks
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Custom filter (REMOVED for Debugging visibility)
    # no_repetitive_logs_filter = NoRepetitiveLogs()
    # logger.addFilter(no_repetitive_logs_filter)
    
    return logger
