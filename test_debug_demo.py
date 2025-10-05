#!/usr/bin/env python3
import logging
import os

# Set LOG_LEVEL environment variable to DEBUG
os.environ["LOG_LEVEL"] = "DEBUG"

# Test logging at different levels
logger = logging.getLogger("demo")

print("Testing with LOG_LEVEL=DEBUG:")
logger.debug("This DEBUG message should be visible")
logger.info("This INFO message should be visible")
logger.warning("This WARNING message should be visible")
logger.error("This ERROR message should be visible")

print("\nTesting current log level:")
print(f"Current root logger level: {logging.getLevelName(logging.root.level)}")
print(f"DEBUG level constant: {logging.DEBUG}")
print(f"INFO level constant: {logging.INFO}")
