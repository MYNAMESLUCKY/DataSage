#!/usr/bin/env python3
"""
Simple API Key Configuration Script
Provides instructions and testing for API keys
"""

import os
import requests

def test_api_key(api_key, name):
    """Test an API key"""
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get("https://api.openrouter.ai/api/v1/models", 
                              headers=headers, timeout=5)
        if response.status_code == 200:
            print(f"✅ {name}: Valid")
            return True
        else:
            print(f"❌ {name}: Invalid (HTTP {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ {name}: Connection failed - {e}")
        return False

def main():
    print("🔑 API Key Configuration for Coding Ground")
    print("=" * 50)
    
    print("\nRequired API Keys:")
    print("1. deepseek_r1_api - OpenRouter key for DeepSeek R1")
    print("2. qwen_api - OpenRouter key for Qwen Coder models")
    print("\nGet your keys from: https://openrouter.ai/keys")
    print("\nTo add keys to Replit:")
    print("1. Open Secrets tab in your Replit project")
    print("2. Add key: deepseek_r1_api with your OpenRouter API key")
    print("3. Add key: qwen_api with your OpenRouter API key")
    print("4. Restart the Coding Ground API workflow")
    
    # Check current environment
    print(f"\n📊 Current Environment Status:")
    deepseek_key = os.getenv("deepseek_r1_api")
    qwen_key = os.getenv("qwen_api")
    
    if deepseek_key:
        print(f"✅ deepseek_r1_api: Set ({deepseek_key[:8]}...{deepseek_key[-4:]})")
        test_api_key(deepseek_key, "DeepSeek API")
    else:
        print("❌ deepseek_r1_api: Not set")
    
    if qwen_key:
        print(f"✅ qwen_api: Set ({qwen_key[:8]}...{qwen_key[-4:]})")
        test_api_key(qwen_key, "Qwen API")
    else:
        print("❌ qwen_api: Not set")
    
    if deepseek_key and qwen_key:
        print(f"\n🎉 Configuration complete! Coding Ground ready to use.")
        print(f"Access: http://localhost:5002")
    else:
        print(f"\n⚠️ Please add the missing API keys to Replit Secrets")

if __name__ == "__main__":
    main()