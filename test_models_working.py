#!/usr/bin/env python3
"""
Test script to verify all AI coding models are working correctly
"""

import requests
import time
import json

def test_coding_models():
    """Test all three AI coding models"""
    
    print("🧪 Coding Ground AI Models Test")
    print("=" * 50)
    
    # Get authentication token
    print("1. Authenticating...")
    auth_response = requests.post('http://localhost:8001/auth/token?user_id=test_user&role=developer')
    
    if auth_response.status_code != 200:
        print("❌ Authentication failed")
        return False
        
    token = auth_response.json()['access_token']
    print("✅ Authentication successful")
    
    # Define models to test
    models = [
        {
            'id': 'deepseek-r1',
            'name': 'DeepSeek R1 🧠',
            'description': 'Advanced reasoning model'
        },
        {
            'id': 'qwen3-coder',
            'name': 'Qwen3 Coder ⚡',
            'description': 'Fast coding assistant'
        },
        {
            'id': 'qwen-coder-32b',
            'name': 'Qwen Coder 32B 🚀',
            'description': 'Large model for complex tasks'
        }
    ]
    
    # Test prompts
    test_prompts = [
        "write a hello world function in python",
        "create a function to calculate factorial",
        "write a simple class for a bank account"
    ]
    
    results = []
    
    for i, model in enumerate(models):
        print(f"\n2.{i+1} Testing {model['name']}...")
        print(f"     {model['description']}")
        
        prompt = test_prompts[i % len(test_prompts)]
        start_time = time.time()
        
        try:
            response = requests.post(
                'http://localhost:8001/code/generate',
                headers={
                    'Authorization': f'Bearer {token}',
                    'Content-Type': 'application/json'
                },
                json={
                    'prompt': prompt,
                    'model': model['id'],
                    'language': 'python',
                    'context': '',
                    'include_docs': True,
                    'search_resources': True
                },
                timeout=90
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    data = result.get('data', {})
                    code = data.get('code', '')
                    explanation = data.get('explanation', '')
                    
                    print(f"     ✅ SUCCESS in {elapsed:.1f}s")
                    print(f"     📝 Code: {len(code)} characters")
                    print(f"     📖 Explanation: {len(explanation)} characters")
                    
                    results.append({
                        'model': model['name'],
                        'status': 'SUCCESS',
                        'time': elapsed,
                        'code_length': len(code),
                        'explanation_length': len(explanation)
                    })
                else:
                    print(f"     ❌ API Error: {result.get('error')}")
                    results.append({
                        'model': model['name'],
                        'status': 'ERROR',
                        'error': result.get('error')
                    })
            else:
                print(f"     ❌ HTTP {response.status_code}")
                results.append({
                    'model': model['name'],
                    'status': 'HTTP_ERROR',
                    'code': response.status_code
                })
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"     ❌ Exception after {elapsed:.1f}s: {e}")
            results.append({
                'model': model['name'],
                'status': 'EXCEPTION',
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    successful = 0
    for result in results:
        if result['status'] == 'SUCCESS':
            successful += 1
            print(f"✅ {result['model']}: {result['time']:.1f}s, {result['code_length']} chars")
        else:
            print(f"❌ {result['model']}: {result['status']}")
    
    print(f"\n🎯 RESULT: {successful}/{len(models)} models working correctly")
    
    if successful == len(models):
        print("🎉 ALL MODELS ARE WORKING PERFECTLY!")
        return True
    else:
        print("⚠️  Some models need attention")
        return False

if __name__ == "__main__":
    test_coding_models()