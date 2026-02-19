"""Tests for the structured logging setup."""

import logging
from pathlib import Path

from src.utils.logger import setup_logger


class TestSetupLogger:
    """Tests for the setup_logger function."""

    def test_creates_logger(self) -> None:
        """Logger is created with the given name."""
        logger = setup_logger("test_logger_create")
        assert logger.name == "test_logger_create"
        assert isinstance(logger, logging.Logger)

    def test_default_level_is_info(self) -> None:
        """Default logging level is INFO."""
        logger = setup_logger("test_level_info")
        assert logger.level == logging.INFO

    def test_custom_level(self) -> None:
        """Custom level string sets the correct numeric level."""
        logger = setup_logger("test_level_debug", level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_console_handler_added(self) -> None:
        """At least one StreamHandler is added to the logger."""
        logger = setup_logger("test_console_handler")
        stream_handlers = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(stream_handlers) >= 1

    def test_file_handler_created(self, tmp_path: Path) -> None:
        """File handler is created when log_file is provided."""
        log_file = tmp_path / "logs" / "test.log"
        logger = setup_logger("test_file_handler", log_file=str(log_file))
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1
        assert log_file.parent.exists()

    def test_idempotent_handler_setup(self) -> None:
        """Calling setup_logger twice doesn't duplicate handlers."""
        name = "test_idempotent"
        logger1 = setup_logger(name)
        handler_count = len(logger1.handlers)
        logger2 = setup_logger(name)
        assert len(logger2.handlers) == handler_count
        assert logger1 is logger2

    def test_log_format(self, tmp_path: Path) -> None:
        """Log output follows the expected structured format."""
        log_file = tmp_path / "format_test.log"
        logger = setup_logger("test_format", log_file=str(log_file))
        logger.info("test message")

        content = log_file.read_text()
        assert "test_format" in content
        assert "INFO" in content
        assert "test message" in content
