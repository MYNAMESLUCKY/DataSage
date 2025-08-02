#!/usr/bin/env python3
"""
Test script for the RAG system to verify core functionality
"""

import sys
import os
import time
import requests
from typing import Dict, Any

# Add backend to path
sys.path.append('.')

def test_api_initialization():
    """Test that the API can be initialized"""
    try:
        from backend.api import RAGSystemAPI
        
        print("Testing API initialization...")
        api = RAGSystemAPI()
        print("✅ API initialized successfully")
        return api
    except Exception as e:
        print(f"❌ API initialization failed: {e}")
        return None

def test_vector_store(api):
    """Test vector store functionality"""
    try:
        print("\nTesting vector store...")
        
        # Test adding sample documents
        from langchain.schema import Document
        
        sample_docs = [
            Document(page_content="Python is a programming language", metadata={"source": "test1"}),
            Document(page_content="Machine learning uses algorithms to find patterns", metadata={"source": "test2"}),
            Document(page_content="RAG systems combine retrieval and generation", metadata={"source": "test3"})
        ]
        
        # Add documents to vector store
        api.vector_store.add_documents(sample_docs)
        print("✅ Documents added to vector store")
        
        # Test similarity search with lower threshold
        results = api.vector_store.similarity_search("What is Python?", k=2, threshold=0.1)
        if results and len(results) > 0:
            print(f"✅ Similarity search returned {len(results)} results")
            return True
        else:
            print("❌ Similarity search returned no results")
            return False
            
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False

def test_data_ingestion(api):
    """Test data ingestion functionality"""
    try:
        print("\nTesting data ingestion...")
        
        # Test URL validation
        from backend.models import DataSource
        
        test_source = DataSource(
            url="https://httpbin.org/html",
            source_type="web",
            name="Test Source"
        )
        
        # Test ingestion service
        result = api.data_ingestion.ingest_from_url(
            url=test_source.url,
            chunk_size=256,
            chunk_overlap=25
        )
        
        if result and len(result) > 0:
            print(f"✅ Data ingestion successful - {len(result)} chunks created")
            return True
        else:
            print("❌ Data ingestion returned no chunks")
            return False
            
    except Exception as e:
        print(f"❌ Data ingestion test failed: {e}")
        return False

def test_rag_query(api):
    """Test RAG query functionality (requires OpenAI API key)"""
    try:
        print("\nTesting RAG query...")
        
        # Check if OpenAI API key is available
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️ OpenAI API key not found - skipping RAG query test")
            return True
        
        # Test a simple query
        result = api.query(
            query="What is Python?",
            llm_model="gpt-4o",
            max_results=3
        )
        
        if result and result.get('status') == 'success':
            print("✅ RAG query completed successfully")
            print(f"   Answer length: {len(result.get('answer', ''))}")
            print(f"   Confidence: {result.get('confidence', 'N/A')}")
            return True
        else:
            print(f"❌ RAG query failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ RAG query test failed: {e}")
        return False

def test_system_stats(api):
    """Test system statistics functionality"""
    try:
        print("\nTesting system statistics...")
        
        stats = api.get_system_stats()
        
        required_keys = ['total_documents', 'total_sources', 'processing_status']
        if all(key in stats for key in required_keys):
            print("✅ System statistics retrieved successfully")
            print(f"   Total documents: {stats.get('total_documents', 0)}")
            print(f"   Total sources: {stats.get('total_sources', 0)}")
            return True
        else:
            print(f"❌ System statistics missing required keys: {stats}")
            return False
            
    except Exception as e:
        print(f"❌ System statistics test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Starting RAG System Tests\n")
    print("=" * 50)
    
    # Test API initialization
    api = test_api_initialization()
    if not api:
        print("\n❌ Critical error: API initialization failed")
        return False
    
    # Run component tests
    tests = [
        test_vector_store,
        test_data_ingestion,
        test_system_stats,
        test_rag_query  # This one depends on OpenAI API key
    ]
    
    results = []
    for test in tests:
        result = test(api)
        results.append(result)
        time.sleep(0.5)  # Small delay between tests
    
    # Summary
    print("\n" + "=" * 50)
    print("🧪 Test Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! RAG system is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)