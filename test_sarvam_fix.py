"""
Test complex queries that should trigger GPU acceleration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.backend.complexity_classifier import classify_query_complexity
from src.backend.gpu_config import get_gpu_service_recommendations

def test_gpu_required_queries():
    """Test queries that should definitely trigger GPU processing"""
    
    # These are designed to score 0.6+ and trigger GPU processing
    gpu_queries = [
        # Mathematical/Computational (high complexity)
        """
        Develop a comprehensive mathematical framework for optimizing multi-dimensional 
        gradient descent algorithms in distributed neural network architectures while 
        considering quantum computational paradigms, non-linear optimization theory, 
        stochastic approximation methods, and information-theoretic bounds on learning 
        efficiency. Analyze convergence rates across different manifold geometries and 
        provide asymptotic complexity bounds for large-scale implementation scenarios.
        """,
        
        # Multi-domain synthesis (very high complexity)
        """
        Synthesize interdisciplinary research spanning quantum field theory, computational 
        neuroscience, phenomenological philosophy, information geometry, category theory, 
        and emergent complexity science to formulate a unified theoretical framework 
        explaining consciousness emergence in artificial systems. Evaluate mathematical 
        formulations of integrated information theory, orchestrated objective reduction, 
        global workspace dynamics, and predictive processing paradigms while addressing 
        the hard problem of consciousness through rigorous computational modeling.
        """,
        
        # Scientific analysis (high complexity)
        """
        Evaluate the mathematical relationships between quantum entanglement phenomena, 
        computational complexity classes, cryptographic security proofs, and topological 
        quantum computing architectures. Analyze how non-classical correlations impact 
        algorithmic efficiency bounds, consider implications for NP-complete problem 
        solving, and develop theoretical models for quantum advantage in machine learning 
        optimization with consideration of decoherence effects and error correction protocols.
        """
    ]
    
    print("🚀 Testing GPU Acceleration Required Queries")
    print("=" * 70)
    
    for i, query in enumerate(gpu_queries, 1):
        print(f"\n{i}. Testing High-Complexity Query:")
        print(f"   {query.strip()[:120]}...")
        print()
        
        # Test complexity classification
        analysis = classify_query_complexity(query.strip())
        
        print("📊 Complexity Analysis:")
        print(f"   Score: {analysis.score:.3f}")
        print(f"   Level: {analysis.level.value.upper()}")
        print(f"   GPU Recommended: {'✅ YES' if analysis.gpu_recommended else '❌ NO'}")
        print(f"   Estimated Time: {analysis.estimated_compute_time:.1f}s")
        
        # Show top complexity factors
        sorted_factors = sorted(analysis.factors.items(), key=lambda x: x[1], reverse=True)
        print("\n   Top Complexity Factors:")
        for factor, score in sorted_factors[:5]:
            if score > 0:
                print(f"     • {factor.replace('_', ' ').title()}: {score:.3f}")
        
        print(f"\n   Reasoning: {analysis.reasoning.split('.')[0]}...")
        
        if analysis.gpu_recommended:
            # Get GPU recommendations
            recommendations = get_gpu_service_recommendations(analysis.score, query.strip())
            
            print(f"\n🔥 GPU Processing Configuration:")
            print(f"   Query Type: {recommendations['query_type']}")
            print(f"   Recommended Service: {recommendations['recommended_services'][0]['name']}")
            print(f"   Compute Power: {recommendations['recommended_services'][0]['compute_capability']}")
            print(f"   Memory: {recommendations['recommended_services'][0]['memory_gb']}GB")
            print(f"   Processing Time: ~{recommendations['estimated_processing_time']['estimated_minutes']} minutes")
        else:
            print("\n⚠️  WARNING: Query did not trigger GPU acceleration!")
            print("   This query may need additional complexity to reach the 0.6 threshold")
        
        print("\n" + "="*70)
    
    # Find the most complex query for testing
    best_query = None
    best_score = 0
    
    for query in gpu_queries:
        analysis = classify_query_complexity(query.strip())
        if analysis.score > best_score:
            best_score = analysis.score
            best_query = query.strip()
    
    print(f"\n🎯 RECOMMENDED TEST QUERY (Score: {best_score:.3f}):")
    print("Copy this query to the Streamlit interface:")
    print()
    print(f'"{best_query[:200]}..."')
    print()
    print("Expected results:")
    print("✓ Complexity score ≥ 0.6")
    print("✓ GPU processing activation")
    print("✓ Enhanced response quality")
    print("✓ Processing time indicators")

if __name__ == "__main__":
    test_gpu_required_queries()