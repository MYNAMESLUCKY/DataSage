#!/usr/bin/env python3
"""
Test Wikipedia integration functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.api import RAGSystemAPI
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_wikipedia_integration():
    """Test the Wikipedia integration functionality"""
    
    print("🧪 Testing Wikipedia Integration")
    print("=" * 50)
    
    # Initialize RAG system
    print("1. Initializing RAG System...")
    api = RAGSystemAPI()
    
    # Check if Wikipedia service is available
    print("2. Checking Wikipedia service availability...")
    if hasattr(api, 'wikipedia_ingestion') and api.wikipedia_ingestion:
        print("✅ Wikipedia ingestion service is available")
    else:
        print("❌ Wikipedia ingestion service not available")
        return
    
    # Test getting categories
    print("3. Testing category retrieval...")
    try:
        categories = api.wikipedia_ingestion.get_wikipedia_categories(5)
        print(f"✅ Retrieved {len(categories)} categories:")
        for cat in categories[:3]:
            print(f"   • {cat}")
        if len(categories) > 3:
            print(f"   ... and {len(categories) - 3} more")
    except Exception as e:
        print(f"❌ Category retrieval failed: {e}")
        return
    
    # Test ingesting a small sample
    print("4. Testing small Wikipedia sample ingestion...")
    try:
        result = api.ingest_wikipedia_random(5)
        if result['status'] == 'success':
            details = result['details']
            print(f"✅ Successfully ingested {details['successful']} articles")
            print(f"   • Documents created: {details['documents_created']}")
            print(f"   • Failed articles: {details['failed']}")
        else:
            print(f"❌ Ingestion failed: {result.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Wikipedia ingestion failed: {e}")
        return
    
    # Test querying the ingested content
    print("5. Testing query with Wikipedia content...")
    try:
        query_result = api.query("What is artificial intelligence?", max_results=3)
        if query_result['status'] == 'success':
            print("✅ Query successful!")
            print(f"   Answer preview: {query_result['answer'][:200]}...")
            print(f"   Sources found: {len(query_result['sources'])}")
        else:
            print(f"❌ Query failed: {query_result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Query test failed: {e}")
    
    # Get system stats
    print("6. Checking system statistics...")
    try:
        stats = api.get_system_stats()
        print(f"✅ System Status:")
        print(f"   • Total documents: {stats.get('total_documents', 0)}")
        print(f"   • Processing status: {stats.get('processing_status', 'unknown')}")
        print(f"   • Vector store initialized: {stats.get('vector_store_initialized', False)}")
    except Exception as e:
        print(f"❌ Stats retrieval failed: {e}")
    
    print("\n🎉 Wikipedia integration test completed!")

if __name__ == "__main__":
    test_wikipedia_integration()