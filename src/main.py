"""Main entry point for the retail analytics application."""

import sys

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main() -> None:
    """Launch the retail analytics application."""
    logger.info("Starting Retail Analytics")
    logger.info("Use 'streamlit run src/dashboard/app.py' for the dashboard")
    logger.info("Use 'python -m src.cli' for the CLI interface")


if __name__ == "__main__":
    sys.exit(main() or 0)
