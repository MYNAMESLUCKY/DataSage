#!/usr/bin/env python3
"""
API Gateway Client Examples
Demonstrates how to use the Enterprise RAG API Gateway
"""

import requests
import json
import time
from typing import Dict, Any

class RAGAPIClient:
    """Client for interacting with RAG API Gateway"""
    
    def __init__(self, base_url: str = "http://localhost:8000", token: str = None):
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.session = requests.Session()
        
        if token:
            self.session.headers.update({'Authorization': f'Bearer {token}'})
    
    def get_token(self, user_id: str, role: str = "user") -> str:
        """Get authentication token"""
        response = self.session.post(
            f"{self.base_url}/auth/token",
            params={"user_id": user_id, "role": role}
        )
        response.raise_for_status()
        
        token_data = response.json()
        self.token = token_data['access_token']
        self.session.headers.update({'Authorization': f'Bearer {self.token}'})
        return self.token
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def query(self, 
              query: str, 
              max_results: int = 10,
              use_web_search: bool = True,
              llm_model: str = "sarvam-m") -> Dict[str, Any]:
        """Query the knowledge base"""
        data = {
            "query": query,
            "max_results": max_results,
            "use_web_search": use_web_search,
            "llm_model": llm_model
        }
        
        response = self.session.post(f"{self.base_url}/query", json=data)
        response.raise_for_status()
        return response.json()
    
    def ingest_url(self, url: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Ingest content from URL"""
        data = {
            "data_type": "url",
            "content": url,
            "metadata": metadata or {}
        }
        
        response = self.session.post(f"{self.base_url}/ingest", json=data)
        response.raise_for_status()
        return response.json()
    
    def ingest_text(self, text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Ingest text content"""
        data = {
            "data_type": "text",
            "content": text,
            "metadata": metadata or {}
        }
        
        response = self.session.post(f"{self.base_url}/ingest", json=data)
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        response = self.session.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> Dict[str, Any]:
        """List available AI models"""
        response = self.session.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()

def main():
    """Example usage of the RAG API Client"""
    
    # Initialize client
    client = RAGAPIClient("http://localhost:8000")
    
    try:
        # Get authentication token
        print("🔐 Getting authentication token...")
        token = client.get_token("demo_user", "user")
        print(f"✅ Token received: {token[:20]}...")
        
        # Check health
        print("\n🏥 Checking API health...")
        health = client.health_check()
        print(f"✅ API Status: {health['status']}")
        print(f"📊 Components: {health['components']}")
        
        # Query the knowledge base
        print("\n🧠 Querying knowledge base...")
        query_result = client.query(
            query="What is quantum computing?",
            max_results=5,
            use_web_search=True
        )
        
        print(f"✅ Query completed in {query_result['processing_time']:.2f}s")
        print(f"📝 Answer: {query_result['answer'][:200]}...")
        print(f"🎯 Confidence: {query_result['confidence']:.1%}")
        print(f"📚 Sources: {len(query_result['sources'])}")
        
        # Get system stats
        print("\n📊 Getting system statistics...")
        stats = client.get_stats()
        print(f"📄 Total Documents: {stats['total_documents']:,}")
        print(f"⏱️ Avg Response Time: {stats['avg_response_time']:.2f}s")
        
        # List available models
        print("\n🤖 Available AI models...")
        models = client.list_models()
        for model in models['models']:
            print(f"  - {model['name']} ({model['id']}) - {model['status']}")
        
        # Ingest sample text
        print("\n📝 Ingesting sample text...")
        ingest_result = client.ingest_text(
            text="This is a sample document about artificial intelligence and machine learning.",
            metadata={"source": "api_example", "type": "demo"}
        )
        print(f"✅ Text ingestion: {ingest_result['status']}")
        
        print("\n🎉 All API operations completed successfully!")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()