# Coding Ground Setup Guide

## Overview
The Coding Ground is a separate AI-powered coding assistant application that integrates with your Enterprise RAG system. It provides Cursor/Lovable-like features using DeepSeek R1 and Qwen3 Coder models.

## Architecture

### Separate Infrastructure
- **Backend API**: Runs on port 8001 (separate from main RAG system on port 8000)
- **Frontend**: Runs on port 5002 (separate from main RAG UI on port 5000)
- **No Interference**: Completely isolated to prevent any impact on your main RAG system

### AI Models Used
1. **DeepSeek R1** (`deepseek-reasoner`)
   - Advanced reasoning for complex coding problems
   - Uses DEEPSEEK_API environment variable

2. **Qwen3 Coder 7B** (`qwen/qwen-2.5-coder-7b-instruct:free`)
   - Fast, efficient coding assistance
   - Uses OPENROUTER_API environment variable

3. **Qwen3 Coder 14B** (`qwen/qwen-2.5-coder-14b-instruct:free`)
   - Enhanced coding capabilities
   - Uses OPENROUTER_API environment variable

## Features

### Core Capabilities
- ✅ **Code Generation**: Natural language to code conversion
- ✅ **Code Explanation**: Detailed analysis of existing code
- ✅ **Error Fixing**: Intelligent debugging assistance
- ✅ **Real-time Execution**: Test Python code immediately
- ✅ **Documentation Search**: Access to programming documentation
- ✅ **Chat Interface**: Natural conversation with AI models

### Supported Languages
- Python (with execution)
- JavaScript
- TypeScript
- Java
- C++
- Go
- Rust
- HTML/CSS

### Documentation Access
- Official Python documentation
- MDN Web Docs
- Stack Overflow solutions
- GitHub repositories
- Open source resources

## How to Use

### 1. Access from Main RAG System
- Navigate to "💻 Coding Ground" in the main navigation
- Check API status and launch the coding interface

### 2. Direct Access
- Frontend: http://localhost:5002
- Backend API: http://localhost:8001

### 3. Workflow Commands
```bash
# Start backend API
python coding_ground_api.py

# Start frontend
streamlit run coding_ground_app.py --server.port 5002
```

## Integration with Main System

### Navigation Integration
The Coding Ground is integrated into your main Enterprise RAG system navigation:
- Added "💻 Coding Ground" option to navigation menu
- Available to all user roles (Admin, User, Viewer)
- Shows real-time API status

### Launch Options
1. **In-App Launch**: Button opens Coding Ground in new tab
2. **Backend Management**: Start/stop backend API
3. **Status Monitoring**: Real-time health check display

## API Endpoints

### Authentication
- `POST /auth/token` - Get access token

### Code Operations
- `POST /code/generate` - Generate code from prompt
- `POST /code/explain` - Explain existing code
- `POST /code/fix` - Fix code errors

### System
- `GET /health` - Health check
- `GET /models` - List available AI models

## External Deployment

### Ngrok Integration
The system is designed to be easily deployed publicly using ngrok:

```bash
# Install ngrok
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok

# Expose Coding Ground API
ngrok http 8001

# Expose Coding Ground Frontend
ngrok http 5002
```

### Use with External Tools
Once deployed with ngrok, the Coding Ground API can be integrated with:
- Cursor IDE
- VS Code extensions
- Custom applications
- Other development tools

## Security Features

### Authentication
- JWT-based authentication
- Role-based access control
- Session management

### Isolation
- Separate port allocation
- Independent error handling
- No impact on main RAG system

### Code Execution Safety
- Sandboxed Python execution
- Timeout protection (30 seconds)
- Temporary file cleanup

## Troubleshooting

### Common Issues
1. **API Not Starting**: Check environment variables (DEEPSEEK_API, OPENROUTER_API)
2. **Frontend Not Loading**: Ensure port 5002 is available
3. **Model Errors**: Verify API keys are valid
4. **Execution Fails**: Check Python environment and permissions

### Environment Variables Required
- `DEEPSEEK_API`: For DeepSeek R1 model
- `OPENROUTER_API`: For Qwen3 Coder models
- `TAVILY_API_KEY`: For documentation search (optional)

### Status Checking
- Visit http://localhost:8001/health for API status
- Check workflow logs in main system
- Monitor real-time status in main navigation

## Future Enhancements

### Planned Features
- Multi-language execution support
- Git integration
- Code review capabilities
- Collaborative coding sessions
- Plugin system for external tools

### Scalability
- Load balancing for multiple instances
- Distributed execution environment
- Enhanced caching for faster responses

---

## Quick Start Commands

```bash
# 1. Start Coding Ground Backend
python coding_ground_api.py

# 2. Start Coding Ground Frontend
streamlit run coding_ground_app.py --server.port 5002

# 3. Access from main system
# Navigate to "💻 Coding Ground" in main interface

# 4. Direct access
# Frontend: http://localhost:5002
# API: http://localhost:8001
```

The Coding Ground is now fully integrated and ready to use alongside your Enterprise RAG system!