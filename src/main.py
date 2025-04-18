import asyncio
import sys
from loguru import logger
from src.logging.log_config import setup_logging
from src.models.queen import Queen

async def main():
    # Set up logging
    setup_logging("main", "main.log")
    logger.info("Starting AntBot Queen...")
    
    try:
        # Initialize colony
        queen = Queen()
        logger.info("Initializing colony...")
        await queen.initialize()
        
        # Start worker management
        logger.info("Starting worker management...")
        await queen.manage_workers()
        
        # Keep the queen running
        logger.info("Queen is running...")
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal. Stopping colony...")
        await queen.stop_colony()
        logger.info("Colony stopped successfully.")
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
        if 'queen' in locals():
            await queen.stop_colony()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 