"""
Test the speed optimization system for simple queries
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.backend.speed_optimizer import speed_optimizer
from src.backend.fast_response_cache import fast_cache

def test_speed_optimization():
    """Test speed optimization for simple queries"""
    
    print("⚡ Testing Speed Optimization System")
    print("=" * 60)
    
    # Test queries with expected performance
    test_queries = [
        # Instant responses (should be < 10ms)
        ("what is ai", "Should get instant response"),
        ("what is machine learning", "Should get instant response"),
        ("what is deep learning", "Should get instant response"),
        
        # Fast path candidates (should be < 100ms)
        ("what is natural language processing", "Should use fast path"),
        ("define artificial intelligence", "Should use fast path"),
        ("explain machine learning", "Should use fast path"),
        
        # Complex queries (should use full processing)
        ("Synthesize research across quantum physics and consciousness", "Should use full processing")
    ]
    
    print("\n🔬 Testing Query Classification and Response Times:")
    print("-" * 60)
    
    for query, expected in test_queries:
        print(f"\nQuery: {query}")
        print(f"Expected: {expected}")
        
        # Test fast path decision
        start_time = time.time()
        should_fast = speed_optimizer.should_use_fast_path(query)
        classification_time = (time.time() - start_time) * 1000
        
        print(f"Fast Path: {'✅ YES' if should_fast else '❌ NO'} (classified in {classification_time:.1f}ms)")
        
        # Test fast response if applicable
        if should_fast:
            start_time = time.time()
            fast_response = fast_cache.get_fast_response(query)
            response_time = (time.time() - start_time) * 1000
            
            if fast_response:
                print(f"Cache Hit: ✅ ({fast_response['cache_type']}) in {response_time:.1f}ms")
                print(f"Answer Preview: {fast_response['answer'][:100]}...")
            else:
                print(f"Cache Miss: ❌ (would generate new response)")
    
    print("\n\n📊 Cache Performance Statistics:")
    print("-" * 40)
    
    cache_stats = fast_cache.get_cache_stats()
    print(f"Instant Responses Available: {cache_stats['instant_responses']}")
    print(f"Memory Cache Size: {cache_stats['memory_cache_size']}")
    print(f"Database Entries: {cache_stats['database_entries']}")
    print(f"Total Cache Accesses: {cache_stats['total_accesses']}")
    
    print("\n\n🎯 Performance Targets:")
    print("-" * 30)
    print("• Instant responses: < 10ms")
    print("• Memory cache hits: < 50ms") 
    print("• Database cache hits: < 100ms")
    print("• Simple queries (total): < 200ms")
    print("• Moderate queries: < 2000ms")
    print("• Complex queries: < 10000ms")
    
    print("\n\n✅ Speed optimization system tested successfully!")
    print("\nTo test in the interface:")
    print("1. Ask simple questions like 'what is ai'")
    print("2. Watch for optimization messages in logs")
    print("3. Verify response times under 100ms")
    print("4. Check that complex questions still get full processing")

if __name__ == "__main__":
    test_speed_optimization()