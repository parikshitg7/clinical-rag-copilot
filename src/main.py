import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def health_check():
    """A simple function to prove the project is runnable."""
    load_dotenv()
    logger.info("Health check passed. System is ready.")
    return True

if __name__ == "__main__":
    health_check()