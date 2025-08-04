"""
Test the hybrid RAG system with complex questions to verify GPU integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_complex_queries():
    """Test queries that should trigger GPU processing"""
    
    complex_queries = [
        # Mathematical/Scientific (should trigger GPU)
        """
        Analyze the mathematical relationships between quantum entanglement, 
        information theory, and computational complexity theory. How do these 
        interdisciplinary connections impact our understanding of consciousness 
        and the nature of reality? Provide a comprehensive framework that 
        integrates philosophical, physical, and computational perspectives.
        """,
        
        # Multi-domain research (should trigger GPU)
        """
        Synthesize research across neuroscience, artificial intelligence, 
        philosophy of mind, and quantum physics to address the hard problem 
        of consciousness. Evaluate competing theories like Integrated Information 
        Theory, Global Workspace Theory, and Orchestrated Objective Reduction, 
        considering their mathematical foundations and empirical predictions.
        """,
        
        # Simple query (should NOT trigger GPU)
        """
        What is artificial intelligence and how does it work?
        """,
        
        # Moderate complexity (should NOT trigger GPU)
        """
        Compare machine learning and deep learning approaches for natural 
        language processing tasks.
        """
    ]
    
    print("🧪 Testing Complex Queries with GPU Integration")
    print("=" * 60)
    
    for i, query in enumerate(complex_queries, 1):
        print(f"\n{i}. Testing Query:")
        print(f"   {query.strip()[:100]}...")
        
        # Test complexity classification
        from src.backend.complexity_classifier import classify_query_complexity
        analysis = classify_query_complexity(query.strip())
        
        print(f"\n   📊 Complexity Analysis:")
        print(f"      Score: {analysis.score:.3f}")
        print(f"      Level: {analysis.level.value.upper()}")
        print(f"      GPU Recommended: {'✅ YES' if analysis.gpu_recommended else '❌ NO'}")
        print(f"      Estimated Time: {analysis.estimated_compute_time:.1f}s")
        
        # Show reasoning
        if analysis.reasoning:
            print(f"      Reasoning: {analysis.reasoning.split('.')[0]}...")
        
        # GPU service recommendations for complex queries
        if analysis.gpu_recommended:
            from src.backend.gpu_config import get_gpu_service_recommendations
            recommendations = get_gpu_service_recommendations(analysis.score, query.strip())
            
            print(f"\n   🚀 GPU Processing Info:")
            print(f"      Query Type: {recommendations['query_type']}")
            print(f"      Recommended Service: {recommendations['recommended_services'][0]['name']}")
            print(f"      Compute: {recommendations['recommended_services'][0]['compute_capability']}")
            print(f"      Memory: {recommendations['recommended_services'][0]['memory_gb']}GB")

if __name__ == "__main__":
    test_complex_queries()
    
    print("\n\n🎯 Manual Testing Instructions:")
    print("=" * 50)
    print("1. Open the Streamlit interface")
    print("2. Copy and paste these complex queries:")
    print()
    
    print("   🔬 COMPLEX QUERY (should show GPU processing):")
    print('   "Analyze quantum consciousness theories integrating neuroscience,')
    print('   physics, and computational complexity theory with mathematical')
    print('   frameworks for emergent properties in distributed systems."')
    print()
    
    print("   📊 SIMPLE QUERY (should use standard processing):")
    print('   "What is machine learning?"')
    print()
    
    print("3. Look for these indicators in the response:")
    print("   ✅ GPU processing messages in logs")
    print("   ✅ Complexity analysis in response metadata")
    print("   ✅ Enhanced response quality for complex queries")
    print("   ✅ 'GPU processing' field in the response")
    print()
    
    print("4. Check the workflow logs for:")
    print("   📝 'Complex query detected' messages")
    print("   📝 'GPU processing successful' confirmations")
    print("   📝 Complexity scores and reasoning")