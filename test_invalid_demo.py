#!/usr/bin/env python3
import logging
import os

# Set LOG_LEVEL environment variable to an invalid value
os.environ["LOG_LEVEL"] = "INVALID_LEVEL"

# Test logging at different levels
logger = logging.getLogger("demo")

print("Testing with LOG_LEVEL=INVALID_LEVEL (should fall back to INFO with warning):")
logger.debug("This DEBUG message should NOT be visible")
logger.info("This INFO message should be visible")
logger.warning("This WARNING message should be visible")
logger.error("This ERROR message should be visible")

print("\nTesting current log level:")
print(f"Current root logger level: {logging.getLevelName(logging.root.level)}")
print("Should be INFO (fallback for invalid value)")
