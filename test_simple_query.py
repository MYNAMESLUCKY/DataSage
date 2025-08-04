#!/usr/bin/env python3
"""
Test SARVAM API with a simple query to avoid rate limiting
"""

import sys
sys.path.append('.')

from src.backend.rag_engine import RAGEngine
from langchain.schema import Document

def test_simple_sarvam_query():
    """Test SARVAM API with a basic query"""
    print("🧪 TESTING SIMPLE SARVAM QUERY")
    print("=" * 40)
    
    # Initialize engine
    engine = RAGEngine()
    engine.initialize()
    
    print(f"Provider: {engine.api_provider}")
    print(f"Model: {engine.default_model}")
    print(f"Available Models: {engine.available_models}")
    
    # Simple test query
    simple_query = "What is 2 + 2?"
    simple_docs = [
        Document(page_content="Mathematics: Addition is a basic arithmetic operation.", metadata={"source": "test"})
    ]
    
    try:
        print(f"\nTesting: {simple_query}")
        result = engine.generate_answer(simple_query, simple_docs)
        
        print("✅ SUCCESS!")
        print(f"Model used: {result.get('model_used')}")
        print(f"Answer: {result.get('answer', '')[:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if "429" in str(e):
            print("Rate limit - this is expected behavior")
        elif "401" in str(e):
            print("Authentication error - this should NOT happen")
    
    print("\n" + "=" * 40)

if __name__ == "__main__":
    test_simple_sarvam_query()