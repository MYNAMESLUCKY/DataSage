#!/usr/bin/env python3
"""
Test script to verify SARVAM API is working correctly in all components
"""

import os
import sys
sys.path.append('.')

from src.backend.api import RAGSystemAPI
from src.backend.rag_engine import RAGEngine
from src.backend.query_processor import AdvancedQueryProcessor
from src.backend.reranker import AdvancedReranker

def test_sarvam_integration():
    """Test SARVAM API across all components"""
    print("🧪 COMPREHENSIVE SARVAM API TEST")
    print("=" * 50)
    
    # Test 1: Direct RAG Engine
    print("\n1. Testing RAG Engine:")
    engine = RAGEngine()
    engine.initialize()
    print(f"   Provider: {engine.api_provider}")
    print(f"   Model: {engine.default_model}")
    print(f"   Ready: {engine.is_ready}")
    
    if engine.openai_client and hasattr(engine.openai_client, 'base_url'):
        print(f"   Base URL: {engine.openai_client.base_url}")
    
    # Test 2: Query Processor
    print("\n2. Testing Query Processor:")
    processor = AdvancedQueryProcessor(engine)
    print(f"   Model: {processor.model}")
    print(f"   Client Available: {processor.client is not None}")
    
    # Test 3: Reranker
    print("\n3. Testing Reranker:")
    reranker = AdvancedReranker(engine)
    print(f"   Model: {reranker.model}")
    print(f"   Client Available: {reranker.client is not None}")
    
    # Test 4: Full API System
    print("\n4. Testing Full API System:")
    api = RAGSystemAPI()
    print(f"   System Ready: {api.is_ready}")
    print(f"   Engine Provider: {api.rag_engine.api_provider}")
    
    # Test 5: Simple query
    print("\n5. Testing Simple Query:")
    try:
        result = engine.generate_answer("What is 2+2?", [])
        print("   ✅ Query successful!")
        print(f"   Model used: {result.get('model_used', 'unknown')}")
        print(f"   Answer preview: {result.get('answer', '')[:100]}...")
    except Exception as e:
        print(f"   ❌ Query failed: {e}")
        if "401" in str(e) or "authentication" in str(e).lower():
            print("   🔍 This is the authentication error we need to fix!")
    
    print("\n" + "=" * 50)
    return engine, processor, reranker, api

if __name__ == "__main__":
    test_sarvam_integration()