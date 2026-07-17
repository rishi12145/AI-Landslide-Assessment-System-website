import uvicorn
from utils.config import load_config
from utils.logger import setup_logger

def main():
    # Load configuration parameters
    config = load_config()
    logger = setup_logger("app_launcher", config.logging)
    
    logger.info("Initializing Landslide Assessment Core Server...")
    logger.info(f"Hosting on {config.server.host}:{config.server.port}")
    logger.info(f"API documentation available at http://{config.server.host}:{config.server.port}/docs")
    
    uvicorn.run(
        "backend.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.debug,
        log_level=config.logging.level.lower()
    )

if __name__ == "__main__":
    main()
