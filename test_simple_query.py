"""
Test the specific complex query about climate change impacts
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.backend.complexity_classifier import classify_query_complexity
from src.backend.gpu_config import get_gpu_service_recommendations

def test_climate_query():
    """Test the specific climate change query"""
    
    query = """Based on the recent climate reports from the IPCC and policy briefs from the United Nations between 2015 and 2023, what are the predicted economic impacts of delayed climate action on coastal agriculture in Southeast Asia, and how do these predictions compare with actual events like the 2020 Mekong River drought?"""
    
    print("🌍 Testing Climate Change Query for GPU Processing")
    print("=" * 60)
    print(f"Query: {query}")
    print()
    
    # Test complexity classification
    analysis = classify_query_complexity(query)
    
    print("📊 Complexity Analysis Results:")
    print(f"   Score: {analysis.score:.3f}")
    print(f"   Level: {analysis.level.value.upper()}")
    print(f"   GPU Recommended: {'✅ YES' if analysis.gpu_recommended else '❌ NO'}")
    print(f"   Estimated Compute Time: {analysis.estimated_compute_time:.1f} seconds")
    print()
    
    # Show detailed factors
    print("🔍 Complexity Factors:")
    sorted_factors = sorted(analysis.factors.items(), key=lambda x: x[1], reverse=True)
    for factor, score in sorted_factors:
        if score > 0:
            print(f"   • {factor.replace('_', ' ').title()}: {score:.3f}")
    print()
    
    # Show reasoning
    print("💭 Analysis Reasoning:")
    print(f"   {analysis.reasoning}")
    print()
    
    if analysis.gpu_recommended:
        # Get GPU service recommendations
        recommendations = get_gpu_service_recommendations(analysis.score, query)
        
        print("🚀 GPU Processing Recommendations:")
        print(f"   Query Type: {recommendations['query_type']}")
        print(f"   Processing Strategy: Distributed computing recommended")
        print()
        
        print("   Top 3 Recommended Services:")
        for i, service in enumerate(recommendations['recommended_services'][:3], 1):
            print(f"   {i}. {service['name']}")
            print(f"      • Compute: {service['compute_capability']}")
            print(f"      • Memory: {service['memory_gb']}GB")
            print(f"      • Suitability: {service['suitability_score']:.2f}/1.0")
            print()
        
        print("⏱️  Estimated Processing:")
        est_time = recommendations['estimated_processing_time']
        print(f"   • Base Time: {est_time['estimated_seconds']}s ({est_time['estimated_minutes']} minutes)")
        print(f"   • Complexity Factor: {est_time['complexity_factor']:.1f}x")
        print()
        
        print("💾 Resource Requirements:")
        resources = recommendations['resource_requirements']
        print(f"   • Minimum Memory: {resources['min_memory_gb']}GB")
        print(f"   • Recommended Memory: {resources['recommended_memory_gb']}GB")
        print(f"   • Compute Capability: {resources['recommended_compute_capability']}")
        print()
    else:
        print("📱 Standard Processing:")
        print("   This query will use regular API processing")
        print("   No GPU acceleration needed")
        print()
    
    print("🎯 Testing Instructions:")
    print("1. Copy this exact query to the Streamlit interface")
    print("2. Submit the question and monitor the logs")
    print("3. Look for complexity classification messages")
    if analysis.gpu_recommended:
        print("4. Watch for GPU processing activation")
        print("5. Check response for enhanced quality indicators")
    else:
        print("4. Verify standard processing is used")
    print()

if __name__ == "__main__":
    test_climate_query()