#!/usr/bin/env python3
"""
Coding Ground - AI-Powered Coding Assistant
Separate application with Cursor/Lovable-like features
Uses DeepSeek R1 and Qwen3 Coder models with internet documentation access
"""

import streamlit as st
import requests
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import subprocess
import tempfile
import sys

# Configure Streamlit page
st.set_page_config(
    page_title="Coding Ground - AI Coding Assistant",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for coding interface
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .stTextArea textarea {
        font-family: 'Monaco', 'Consolas', 'Courier New', monospace;
        font-size: 14px;
        line-height: 1.4;
        background-color: #1e1e1e;
        color: #d4d4d4;
    }
    .code-editor {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .output-panel {
        background-color: #0d1117;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        color: #58a6ff;
        font-family: monospace;
    }
    .chat-message {
        padding: 10px;
        margin: 5px 0;
        border-radius: 8px;
        border-left: 4px solid #58a6ff;
        background-color: #161b22;
    }
    .model-badge {
        background-color: #238636;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        margin: 2px;
    }
    .feature-card {
        background-color: #21262d;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class CodingGroundAPI:
    """API client for the separate coding ground backend"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def get_auth_token(self, user_id: str = "coding_user") -> Optional[str]:
        """Get authentication token for coding API"""
        try:
            response = self.session.post(f"{self.base_url}/auth/token?user_id={user_id}&role=developer")
            if response.status_code == 200:
                return response.json()["access_token"]
        except:
            pass
        return None
    
    def generate_code(self, prompt: str, model: str, language: str = "python", context: str = "") -> Dict[str, Any]:
        """Generate code using AI models"""
        token = self.get_auth_token()
        if not token:
            return {"error": "Authentication failed"}
            
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "prompt": prompt,
            "model": model,
            "language": language,
            "context": context,
            "include_docs": True,
            "search_resources": True
        }
        
        try:
            response = self.session.post(f"{self.base_url}/code/generate", 
                                       headers=headers, json=payload, timeout=90)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Connection failed: {str(e)}"}
    
    def explain_code(self, code: str, model: str) -> Dict[str, Any]:
        """Explain existing code"""
        token = self.get_auth_token()
        if not token:
            return {"error": "Authentication failed"}
            
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "code": code,
            "model": model,
            "include_docs": True
        }
        
        try:
            response = self.session.post(f"{self.base_url}/code/explain", 
                                       headers=headers, json=payload, timeout=90)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Connection failed: {str(e)}"}
    
    def fix_code(self, code: str, error: str, model: str) -> Dict[str, Any]:
        """Fix code based on error"""
        token = self.get_auth_token()
        if not token:
            return {"error": "Authentication failed"}
            
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "code": code,
            "error": error,
            "model": model,
            "search_docs": True
        }
        
        try:
            response = self.session.post(f"{self.base_url}/code/fix", 
                                       headers=headers, json=payload, timeout=90)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Connection failed: {str(e)}"}

def execute_code(code: str, language: str) -> Dict[str, Any]:
    """Execute code safely in temporary environment"""
    try:
        if language.lower() == "python":
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Execute Python code
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up
            os.unlink(temp_file)
            
            return {
                "output": result.stdout,
                "error": result.stderr,
                "returncode": result.returncode,
                "success": result.returncode == 0
            }
        else:
            return {"error": "Language not supported for execution"}
            
    except subprocess.TimeoutExpired:
        return {"error": "Code execution timed out"}
    except Exception as e:
        return {"error": f"Execution failed: {str(e)}"}

def main():
    """Main Coding Ground application"""
    
    st.title("💻 Coding Ground")
    st.markdown("**AI-Powered Coding Assistant with Documentation Access**")
    
    # Initialize API client
    api = CodingGroundAPI()
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_code' not in st.session_state:
        st.session_state.current_code = ""
    if 'execution_output' not in st.session_state:
        st.session_state.execution_output = ""
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("🤖 AI Models")
        
        # Model info
        model_info = {
            "deepseek-r1": {
                "name": "DeepSeek R1 🧠",
                "description": "Advanced reasoning for complex problems",
                "best_for": "Complex algorithms, architecture design, debugging"
            },
            "qwen3-coder": {
                "name": "Qwen3 Coder ⚡",
                "description": "Latest fast coding assistant",
                "best_for": "Quick tasks, code completion, simple functions"
            },
            "qwen-coder-32b": {
                "name": "Qwen Coder 32B 🚀",
                "description": "Large model for complex coding tasks",
                "best_for": "Full applications, refactoring, optimization"
            }
        }
        
        selected_model = st.selectbox(
            "Choose your AI coding model:",
            list(model_info.keys()),
            format_func=lambda x: model_info[x]["name"],
            help="Different models excel at different coding tasks"
        )
        
        # Show model details
        model_details = model_info[selected_model]
        st.markdown(f"""
        <div class="feature-card">
        <strong>{model_details['name']}</strong><br>
        <em>{model_details['description']}</em><br><br>
        <strong>Best for:</strong> {model_details['best_for']}
        </div>
        """, unsafe_allow_html=True)
        
        # Language selection
        selected_language = st.selectbox(
            "Programming Language:",
            ["Python", "JavaScript", "TypeScript", "Java", "C++", "Go", "Rust", "HTML/CSS"],
            index=0
        )
        
        # Quick model switching
        st.markdown("**Quick Switch:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🧠 R1", help="DeepSeek R1 - Advanced reasoning"):
                selected_model = "deepseek-r1"
                st.rerun()
        with col2:
            if st.button("⚡ Q3", help="Qwen3 Coder - Fast coding"):
                selected_model = "qwen3-coder"
                st.rerun()
        with col3:
            if st.button("🚀 32B", help="Qwen 32B - Advanced coding"):
                selected_model = "qwen-coder-32b"
                st.rerun()
        
        # Features
        st.subheader("🚀 Features")
        st.markdown("""
        <div class="feature-card">
        <strong>Available Features:</strong><br>
        ✓ Multiple AI Models<br>
        ✓ Code Generation<br>
        ✓ Code Explanation<br>
        ✓ Error Fixing<br>
        ✓ Documentation Search<br>
        ✓ Real-time Execution<br>
        ✓ Chat Interface<br>
        </div>
        """, unsafe_allow_html=True)
        
        # API Status
        st.subheader("📡 Status")
        try:
            test_response = requests.get("http://localhost:8001/health", timeout=2)
            if test_response.status_code == 200:
                st.success("✅ Coding API Online")
                
                # Test a quick model call to check rate limits
                auth_response = requests.post("http://localhost:8001/auth/token?user_id=status_check&role=developer", timeout=2)
                if auth_response.status_code == 200:
                    token = auth_response.json()["access_token"]
                    model_test = requests.post(
                        "http://localhost:8001/code/generate",
                        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                        json={"prompt": "test", "model": "deepseek-r1", "language": "python"},
                        timeout=5
                    )
                    if model_test.status_code == 200:
                        result = model_test.json()
                        if "rate limit" in result.get("data", {}).get("explanation", "").lower():
                            st.warning("⚠️ API Rate Limit Reached")
                            st.info("Using smart fallback responses. Resets daily.")
                        else:
                            st.success("🚀 All Models Available")
                    else:
                        st.info("🔄 Model Status Unknown")
            else:
                st.error("❌ Coding API Error")
        except:
            st.warning("⚠️ Coding API Offline")
            st.info("Starting coding backend...")
    
    # Main layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("💬 AI Chat Interface")
        
        # Chat input
        user_input = st.text_area(
            "Ask me anything about coding:",
            height=100,
            placeholder="e.g., 'Create a REST API with FastAPI' or 'Explain this code' or 'Fix this error'"
        )
        
        # Action buttons
        col1a, col1b, col1c = st.columns(3)
        
        with col1a:
            if st.button("🤖 Generate Code", use_container_width=True):
                if user_input:
                    with st.spinner("Generating code..."):
                        result = api.generate_code(
                            prompt=user_input,
                            model=selected_model,
                            language=selected_language.lower(),
                            context=st.session_state.current_code
                        )
                        
                        if "error" in result:
                            st.error(f"Error: {result['error']}")
                        else:
                            st.session_state.current_code = result.get("code", "")
                            st.session_state.chat_history.append({
                                "type": "request",
                                "content": user_input,
                                "model": selected_model,
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                            st.session_state.chat_history.append({
                                "type": "response",
                                "content": result.get("explanation", "Code generated successfully"),
                                "code": result.get("code", ""),
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                            st.rerun()
        
        with col1b:
            if st.button("📖 Explain Code", use_container_width=True):
                if st.session_state.current_code:
                    with st.spinner("Analyzing code..."):
                        result = api.explain_code(
                            code=st.session_state.current_code,
                            model=selected_model
                        )
                        
                        if "error" in result:
                            st.error(f"Error: {result['error']}")
                        else:
                            st.session_state.chat_history.append({
                                "type": "explanation",
                                "content": result.get("explanation", "Code explained"),
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                            st.rerun()
        
        with col1c:
            if st.button("🔧 Fix Errors", use_container_width=True):
                if st.session_state.execution_output and "error" in st.session_state.execution_output.lower():
                    with st.spinner("Fixing code..."):
                        result = api.fix_code(
                            code=st.session_state.current_code,
                            error=st.session_state.execution_output,
                            model=selected_model
                        )
                        
                        if "error" in result:
                            st.error(f"Error: {result['error']}")
                        else:
                            st.session_state.current_code = result.get("fixed_code", st.session_state.current_code)
                            st.session_state.chat_history.append({
                                "type": "fix",
                                "content": result.get("explanation", "Code fixed"),
                                "code": result.get("fixed_code", ""),
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                            st.rerun()
        
        # Chat history
        st.subheader("💭 Chat History")
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_history[-10:]:  # Show last 10 messages
                if message["type"] == "request":
                    st.markdown(f"""
                    <div class="chat-message">
                    <strong>🧑 You ({message['timestamp']}):</strong><br>
                    {message['content']}<br>
                    <span class="model-badge">Model: {message['model']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                elif message["type"] in ["response", "explanation", "fix"]:
                    icon = "🤖" if message["type"] == "response" else "📖" if message["type"] == "explanation" else "🔧"
                    st.markdown(f"""
                    <div class="chat-message">
                    <strong>{icon} AI ({message['timestamp']}):</strong><br>
                    {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
    
    with col2:
        st.header("📝 Code Editor")
        
        # Code editor
        st.session_state.current_code = st.text_area(
            f"{selected_language} Code:",
            value=st.session_state.current_code,
            height=300,
            help="Edit your code here"
        )
        
        # Execution buttons
        col2a, col2b, col2c = st.columns(3)
        
        with col2a:
            if st.button("▶️ Run Code", use_container_width=True):
                if st.session_state.current_code:
                    with st.spinner("Executing code..."):
                        result = execute_code(st.session_state.current_code, selected_language)
                        
                        if result.get("success"):
                            st.session_state.execution_output = result.get("output", "")
                            st.success("✅ Code executed successfully")
                        else:
                            st.session_state.execution_output = result.get("error", "Unknown error")
                            st.error("❌ Execution failed")
                        st.rerun()
        
        with col2b:
            if st.button("💾 Save Code", use_container_width=True):
                if st.session_state.current_code:
                    # Create download link
                    filename = f"code_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{selected_language.lower()}"
                    st.download_button(
                        label="📥 Download",
                        data=st.session_state.current_code,
                        file_name=filename,
                        mime="text/plain"
                    )
        
        with col2c:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.current_code = ""
                st.session_state.execution_output = ""
                st.rerun()
        
        # Output panel
        st.subheader("📤 Output")
        if st.session_state.execution_output:
            st.markdown(f"""
            <div class="output-panel">
            <pre>{st.session_state.execution_output}</pre>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Run your code to see output here")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    <strong>Coding Ground</strong> - AI-Powered Development Environment<br>
    Powered by DeepSeek R1 & Qwen3 Coder • Documentation Access • Open Source Resources
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()