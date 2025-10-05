"""
PyTorch Device Configuration Module
==================================
Centralized device management for PyTorch operations across the application.
"""

import argparse
import logging
import threading
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class DeviceConfig:
    """Centralized PyTorch device configuration and management."""

    def __init__(self, device_str: Optional[str] = None):
        """
        Initialize device configuration.

        Args:
            device_str: Device specification like 'cpu', 'cuda', 'cuda:0', etc.
                       If None, auto-detects best available device.
        """
        self.device = self._configure_device(device_str)
        self._setup_thread_configuration()

        logger.info(f"PyTorch device configured: {self.device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")

    def _configure_device(self, device_str: Optional[str]) -> torch.device:
        """Configure and validate the PyTorch device."""
        if device_str is None:
            # Auto-detect best available device
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("Auto-detected device: CUDA")
            else:
                device = torch.device("cpu")
                logger.info("Auto-detected device: CPU")
        else:
            device = self._parse_and_validate_device(device_str)

        return device

    @staticmethod
    def _parse_and_validate_device(device_str: str) -> torch.device:
        """Parse and validate the device string."""
        try:
            device = torch.device(device_str)

            # Validate CUDA devices
            if device.type == "cuda":
                if not torch.cuda.is_available():
                    logger.warning(
                        f"CUDA requested ({device_str}) but not available. Falling back to CPU."
                    )
                    return torch.device("cpu")

                # Check specific CUDA device index
                if device.index is not None:
                    if device.index >= torch.cuda.device_count():
                        logger.warning(
                            f"CUDA device {device.index} not available "
                            f"(only {torch.cuda.device_count()} devices). Falling back to CPU."
                        )
                        return torch.device("cpu")

                    # Test device accessibility
                    try:
                        torch.zeros(1, device=device)
                        logger.info(f"Validated CUDA device: {device}")
                    except RuntimeError as e:
                        logger.warning(
                            f"CUDA device {device} not accessible: {e}. Falling back to CPU."
                        )
                        return torch.device("cpu")
                else:
                    # Default to cuda:0 if just 'cuda' specified
                    device = torch.device("cuda:0")
                    try:
                        torch.zeros(1, device=device)
                        logger.info(f"Validated CUDA device: {device}")
                    except RuntimeError as e:
                        logger.warning(
                            f"Default CUDA device not accessible: {e}. Falling back to CPU."
                        )
                        return torch.device("cpu")

            return device

        except RuntimeError as e:
            logger.error(f"Invalid device specification '{device_str}': {e}")
            logger.info("Falling back to CPU")
            return torch.device("cpu")

    def _setup_thread_configuration(self):
        """Configure PyTorch threading based on device selection."""
        if self.device.type == "cuda":
            # Set single thread when using CUDA to prevent oversubscription
            torch.set_num_threads(1)
            logger.info(
                "Set PyTorch to single thread mode for CUDA device to prevent thread oversubscription"
            )
        else:
            # For CPU, let PyTorch use default threading
            logger.info(
                f"Using PyTorch default threading for CPU (threads: {torch.get_num_threads()})"
            )

    def get_device(self) -> torch.device:
        """Get the configured device."""
        return self.device

    def to_device(self, tensor_or_model):
        """Move tensor or model to the configured device."""
        return tensor_or_model.to(self.device)

    def get_device_info(self) -> dict:
        """Get device information for logging/debugging."""
        info = {
            "device_str": str(self.device),
            "device_type": self.device.type,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "num_threads": torch.get_num_threads(),
        }

        if torch.cuda.is_available():
            info.update(
                {
                    "cuda_device_count": torch.cuda.device_count(),
                    "cuda_version": torch.version.cuda,
                    "cudnn_version": torch.backends.cudnn.version(),
                }
            )

            if self.device.type == "cuda":
                device_idx = self.device.index or 0
                if device_idx < torch.cuda.device_count():
                    info.update(
                        {
                            "device_name": torch.cuda.get_device_name(device_idx),
                            "device_capability": torch.cuda.get_device_capability(
                                device_idx
                            ),
                            "memory_allocated": torch.cuda.memory_allocated(device_idx),
                            "memory_cached": torch.cuda.memory_reserved(device_idx),
                        }
                    )

        return info


# Global device configuration instance (lazy initialised via get_device_config)
_device_config: Optional[DeviceConfig] = None
_device_config_lock = threading.Lock()


def get_device_config() -> DeviceConfig:
    """Return the shared :class:`DeviceConfig`, creating it on first use.

    The instance is created lazily with a thread-safe double-checked locking
    pattern so concurrent initialisation attempts do not race. The returned
    object should be treated as immutable after creation.
    """

    global _device_config
    if _device_config is None:
        with _device_config_lock:
            if _device_config is None:
                _device_config = DeviceConfig()
    return _device_config


def initialize_device_config(device_str: Optional[str] = None) -> DeviceConfig:
    """Initialize the global device configuration."""
    config = DeviceConfig(device_str)
    replace_device_config(config)
    return config


def replace_device_config(config: Optional[DeviceConfig]) -> None:
    """Replace the shared device configuration (intended for tests).

    This helper allows tests to inject deterministic configurations while the
    production code treats the singleton as effectively immutable.
    """

    global _device_config
    with _device_config_lock:
        _device_config = config


def get_device() -> torch.device:
    """Get the configured PyTorch device."""
    return get_device_config().get_device()


def to_device(tensor_or_model):
    """Move tensor or model to the configured device."""
    return get_device_config().to_device(tensor_or_model)


def add_device_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add device-related command line arguments to an ArgumentParser."""
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='PyTorch device to use (e.g., "cpu", "cuda", "cuda:0"). Auto-detects if not specified.',
    )
    return parser


def configure_device_from_args(args) -> DeviceConfig:
    """Configure device from parsed command line arguments."""
    return initialize_device_config(getattr(args, "device", None))
