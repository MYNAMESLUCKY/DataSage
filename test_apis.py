#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()
import os
import requests

def check_tavily():
    key = os.getenv('TAVILY_API_KEY')
    if not key:
        return 'TAVILY_API_KEY not found'
    try:
        # Remove quotes from key value if present
        key = key.strip('"')
        headers = {
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json'
        }
        
        # Use current news query to ensure fresh results
        data = {
            "query": "What are Messi's most recent achievements in 2025?",
            "search_depth": "advanced",  # Get more comprehensive results
            "include_answer": True,
            "include_raw_content": True,
            "max_results": 10,  # Get more results
            "search_on": "google,bing",  # Use multiple search engines
            "api_feature_flags": {
                "use_cache": False  # Disable cache to get fresh results
            }
        }
        
        r = requests.post('https://api.tavily.com/search',
                         json=data,
                         headers=headers,
                         timeout=10)
        
        # Format response for better readability
        if r.status_code == 200:
            try:
                json_response = r.json()
                result = f'Tavily API: {r.status_code} - Search successful\n'
                
                # Add answer if available
                if json_response.get("answer"):
                    result += f'\nAnswer: {json_response["answer"][:300]}...\n'
                
                # Add search results
                if json_response.get("results"):
                    result += "\nTop 3 Search Results:\n"
                    for idx, res in enumerate(json_response["results"][:3], 1):
                        result += f"\n{idx}. {res.get('title', 'No title')}"
                        result += f"\nURL: {res.get('url', 'No URL')}"
                        result += f"\nContent: {res.get('content', 'No content')[:200]}...\n"
                
                return result
            except Exception as e:
                return f'Tavily API: {r.status_code} - Error parsing response: {str(e)}\n{r.text[:200]}'
        else:
            return f'Tavily API: {r.status_code} - {r.text[:200]}'
    except Exception as e:
        return f'Tavily API Error: {str(e)}'

def check_sarvam():
    key = os.getenv('SARVAM_API')
    if not key:
        return 'SARVAM_API not found'
    try:
        # Remove quotes from key value if present
        key = key.strip('"')
        headers = {
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json'
        }
        data = {
            'messages': [{'role': 'user', 'content': 'hello'}],
            'temperature': 0.7,
            'model': 'sarvam-m'  # Updated to correct model name
        }
        r = requests.post('https://api.sarvam.ai/v1/chat/completions', 
                         headers=headers, json=data, timeout=5)
        response_text = r.text[:200]  # Get first 200 chars of response
        return f'Sarvam API: {r.status_code} - {response_text}'
    except Exception as e:
        return f'Sarvam API Error: {str(e)}'

if __name__ == '__main__':
    print('Checking API Keys...')
    print(check_tavily())
    print(check_sarvam())
