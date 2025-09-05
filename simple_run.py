#!/usr/bin/env python3
"""
Simple Compliance System Runner - Development Mode
Starts the server with minimal dependencies to avoid compatibility issues
"""
import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Simple main entry point for development"""
    try:
        print("🚀 Starting Compliance System (Development Mode)")
        print("📁 Working directory:", os.getcwd())
        
        # Set environment variables for safe startup
        os.environ['SKIP_VECTOR_STORE'] = 'true'
        os.environ['MOCK_LLM'] = 'true'
        os.environ['DEBUG'] = 'true'
        os.environ['API_HOST'] = '127.0.0.1'
        os.environ['API_PORT'] = '8000'
        
        # Import and start uvicorn directly
        import uvicorn
        
        print("✅ Dependencies loaded successfully")
        print("🌐 Starting server at http://127.0.0.1:8000")
        print("📚 API docs will be available at http://127.0.0.1:8000/docs")
        print("❤️  Health check at http://127.0.0.1:8000/health")
        print("")
        print("Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Start the server with minimal configuration
        uvicorn.run(
            "app.main:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True,
            workers=1,
            loop="asyncio"
        )
        
    except KeyboardInterrupt:
        print("\n👋 Server shutdown requested")
        print("✅ Compliance System stopped")
        
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure you're in the virtual environment: source venv/bin/activate")
        print("2. Install missing dependencies: pip install -r requirements.txt")
        print("3. Check if port 8000 is available: lsof -i :8000")
        return 1

if __name__ == "__main__":
    sys.exit(main() or 0)