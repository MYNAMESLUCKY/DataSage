#!/usr/bin/env python3
"""
Test script for the intelligent hybrid RAG system
"""

import sys
import os
sys.path.append('src')

# Add the current directory to the path
sys.path.insert(0, os.path.abspath('.'))

try:
    from src.backend.hybrid_rag_processor import HybridRAGProcessor
    from src.backend.api import RAGSystemAPI
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Trying alternative import method...")
    try:
        import src.backend.hybrid_rag_processor as hybrid_module
        import src.backend.api as api_module
        HybridRAGProcessor = hybrid_module.HybridRAGProcessor
        RAGSystemAPI = api_module.RAGSystemAPI
        print("✅ Alternative imports successful")
    except Exception as e2:
        print(f"❌ All imports failed: {e2}")
        sys.exit(1)

def test_hybrid_system():
    """Test the hybrid RAG system with a complex query"""
    
    print("🔄 Initializing RAG System...")
    try:
        # Initialize the API
        api = RAGSystemAPI()
        print("✅ RAG System API initialized")
        
        # Initialize hybrid processor
        hybrid_processor = HybridRAGProcessor(
            api.vector_store,
            api.rag_engine,
            api.enhanced_retrieval
        )
        print("✅ Hybrid processor initialized")
        
        # Test query
        test_query = "What are the latest developments in quantum computing for agricultural optimization?"
        print(f"\n🧠 Testing query: {test_query}")
        
        # Process with hybrid system
        result = hybrid_processor.process_intelligent_query(
            query=test_query,
            llm_model="sarvam-m",
            use_web_search=True,
            max_web_results=3
        )
        
        print(f"\n📊 Results:")
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Strategy: {result.get('strategy_used', 'unknown')}")
        print(f"KB Documents: {result.get('kb_documents_found', 0)}")
        print(f"Web Results: {result.get('web_results_used', 0)}")
        print(f"KB Updated: {result.get('knowledge_base_updated', False)}")
        print(f"Processing Time: {result.get('processing_time', 0):.2f}s")
        
        if result.get('status') == 'success':
            print(f"\n💡 Answer Preview: {result.get('answer', '')[:200]}...")
            print(f"Web Sources: {len(result.get('web_sources', []))}")
            print(f"Insights: {result.get('insights', 'None')}")
            print("\n✅ Test completed successfully!")
            return True
        else:
            print(f"\n❌ Test failed: {result.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Intelligent Hybrid RAG System")
    print("=" * 50)
    success = test_hybrid_system()
    sys.exit(0 if success else 1)