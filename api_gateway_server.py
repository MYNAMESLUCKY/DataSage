#!/usr/bin/env python3
"""
Standalone API Gateway Server
Run with: python api_gateway_server.py
"""

import uvicorn
import os
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    # Configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    
    print(f"🚀 Starting Enterprise RAG API Gateway")
    print(f"📡 Server: http://{host}:{port}")
    print(f"📚 Documentation: http://{host}:{port}/docs")
    print(f"🔧 Interactive API: http://{host}:{port}/redoc")
    
    # Run the FastAPI application
    uvicorn.run(
        "api.gateway:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True
    )