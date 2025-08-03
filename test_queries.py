"""
Test script to validate RAG system improvements
==============================================

This script tests the 8-point improvement system with various query types.
"""

import sys
import os
import time

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.api import RAGSystemAPI

def test_rag_system():
    """Test the enhanced RAG system with various queries"""
    
    print("🧠 Testing Enhanced RAG System")
    print("=" * 50)
    
    # Initialize API
    api = RAGSystemAPI()
    
    # Test queries grouped by type
    test_queries = {
        "Basic Definitions": [
            "what is water",
            "what is artificial intelligence", 
            "define photosynthesis"
        ],
        "Historical Questions": [
            "when did World War II start and end",
            "who was the first person to walk on the moon"
        ],
        "Scientific Questions": [
            "how does gravity work",
            "explain the water cycle"
        ],
        "Complex Multi-Concept": [
            "how does climate change affect ocean temperatures",
            "relationship between DNA and protein synthesis"
        ]
    }
    
    results = {}
    
    for category, queries in test_queries.items():
        print(f"\n📋 Testing {category}")
        print("-" * 30)
        
        category_results = []
        
        for query in queries:
            print(f"\n🔍 Query: {query}")
            
            start_time = time.time()
            result = api.query(
                query=query,
                max_results=10,
                similarity_threshold=0.15,
                use_cache=False
            )
            end_time = time.time()
            
            # Analyze result
            if result['status'] == 'success':
                print(f"✅ Success!")
                print(f"   📊 Confidence: {result.get('confidence', 0):.2f}")
                print(f"   ⏱️  Time: {result.get('processing_time', end_time-start_time):.2f}s")
                print(f"   📄 Sources: {len(result.get('sources', []))}")
                print(f"   🔄 Fallback Used: {result.get('fallback_used', False)}")
                
                # Show first 200 chars of answer
                answer = result.get('answer', '')
                preview = answer[:200] + "..." if len(answer) > 200 else answer
                print(f"   💬 Answer Preview: {preview}")
                
                category_results.append({
                    'query': query,
                    'success': True,
                    'confidence': result.get('confidence', 0),
                    'response_time': result.get('processing_time', end_time-start_time),
                    'sources_count': len(result.get('sources', [])),
                    'fallback_used': result.get('fallback_used', False),
                    'answer_length': len(answer)
                })
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                category_results.append({
                    'query': query,
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                })
        
        results[category] = category_results
    
    # Generate summary report
    print("\n" + "="*50)
    print("📊 TEST SUMMARY REPORT")
    print("="*50)
    
    total_queries = 0
    successful_queries = 0
    total_time = 0
    total_confidence = 0
    fallback_count = 0
    
    for category, category_results in results.items():
        print(f"\n📋 {category}:")
        
        category_successful = sum(1 for r in category_results if r.get('success', False))
        category_total = len(category_results)
        
        print(f"   Success Rate: {category_successful}/{category_total} ({category_successful/category_total*100:.1f}%)")
        
        if category_successful > 0:
            avg_confidence = sum(r.get('confidence', 0) for r in category_results if r.get('success', False)) / category_successful
            avg_time = sum(r.get('response_time', 0) for r in category_results if r.get('success', False)) / category_successful
            category_fallback = sum(1 for r in category_results if r.get('fallback_used', False))
            
            print(f"   Avg Confidence: {avg_confidence:.2f}")
            print(f"   Avg Response Time: {avg_time:.2f}s")
            print(f"   Fallback Usage: {category_fallback}/{category_successful}")
            
            total_confidence += avg_confidence * category_successful
            total_time += avg_time * category_successful
            fallback_count += category_fallback
        
        total_queries += category_total
        successful_queries += category_successful
    
    # Overall statistics
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   Total Success Rate: {successful_queries}/{total_queries} ({successful_queries/total_queries*100:.1f}%)")
    
    if successful_queries > 0:
        print(f"   Overall Avg Confidence: {total_confidence/successful_queries:.2f}")
        print(f"   Overall Avg Response Time: {total_time/successful_queries:.2f}s")
        print(f"   Overall Fallback Usage: {fallback_count}/{successful_queries} ({fallback_count/successful_queries*100:.1f}%)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if successful_queries < total_queries:
        print("   • Some queries failed - check data coverage")
    
    if successful_queries > 0:
        avg_confidence = total_confidence / successful_queries
        if avg_confidence < 0.7:
            print("   • Low confidence scores - consider improving document quality")
        
        avg_time = total_time / successful_queries
        if avg_time > 5.0:
            print("   • Slow response times - consider optimizing retrieval")
        
        fallback_rate = fallback_count / successful_queries
        if fallback_rate > 0.3:
            print("   • High fallback usage - expand knowledge base")
        
        if avg_confidence > 0.8 and avg_time < 3.0 and fallback_rate < 0.2:
            print("   🎉 Excellent performance! All metrics are in good ranges.")
    
    return results

if __name__ == "__main__":
    test_rag_system()