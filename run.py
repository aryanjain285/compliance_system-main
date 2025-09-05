#!/usr/bin/env python3
"""
Compliance System Runner
Production-ready entry point for the compliance system
"""
import os
import sys
import uvicorn
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.config.settings import get_settings
from app.utils.logger import setup_logging, get_logger

def main():
    """Main entry point for the application"""
    # Setup logging first
    setup_logging()
    logger = get_logger(__name__)
    
    # Get settings
    settings = get_settings()
    
    # Create required directories
    try:
        settings.create_upload_dir()
        settings.create_log_dir()
        
        # Create chroma directory if vector store is enabled
        if not settings.skip_vector_store:
            os.makedirs(settings.chroma_persist_dir, exist_ok=True)
            
    except Exception as e:
        logger.error(f"Failed to create required directories: {e}")
        sys.exit(1)
    
    logger.info("Starting Compliance System API Server")
    logger.info(f"Environment: {'Production' if settings.is_production else 'Development'}")
    logger.info(f"Host: {settings.api_host}:{settings.api_port}")
    logger.info(f"Log Level: {settings.log_level}")
    logger.info(f"Database: {settings.database_url}")
    
    # Feature flags
    logger.info(f"Features enabled:")
    logger.info(f"  - LLM Service: {bool(settings.openai_api_key or settings.anthropic_api_key)}")
    logger.info(f"  - Vector Store: {not settings.skip_vector_store}")
    logger.info(f"  - File Upload: True")
    logger.info(f"  - Email Notifications: {settings.has_email_config}")
    
    try:
        # Run the server
        uvicorn.run(
            "app.main:app",
            host=settings.api_host,
            port=settings.api_port,
            reload=settings.api_reload,
            log_level=settings.log_level.lower(),
            access_log=settings.is_development,
            workers=1,  # Always use 1 worker to avoid multiprocessing issues
            loop="asyncio",
            http="h11",
        )
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()