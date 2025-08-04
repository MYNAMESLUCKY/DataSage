"""
Query Complexity Classifier
Analyzes incoming queries to determine if they require GPU acceleration or can be handled by standard APIs.
"""

import re
import logging
from typing import Dict, List, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)

class QueryComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    RESEARCH = "research"

class QueryClassifier:
    """Classifies queries based on complexity to determine processing approach"""
    
    def __init__(self):
        # Keywords that indicate complex queries requiring GPU acceleration
        self.complex_keywords = {
            'technical': ['algorithm', 'architecture', 'implementation', 'optimization', 'performance', 'scalability'],
            'research': ['analyze', 'compare', 'research', 'investigate', 'comprehensive', 'detailed analysis'],
            'multi_step': ['step by step', 'explain how', 'walk through', 'process of', 'methodology'],
            'academic': ['theorem', 'hypothesis', 'dissertation', 'academic', 'scholarly', 'peer-reviewed'],
            'advanced': ['quantum', 'machine learning', 'artificial intelligence', 'blockchain', 'cryptography'],
            'business': ['strategy', 'framework', 'assessment', 'roi', 'business case', 'market analysis']
        }
        
        # Indicators of simple queries that can use standard APIs
        self.simple_indicators = [
            'what is', 'define', 'explain', 'basic', 'simple', 'introduction',
            'summary', 'overview', 'meaning', 'definition'
        ]
        
        # Query length thresholds
        self.length_thresholds = {
            'short': 50,    # Simple queries
            'medium': 200,  # Moderate complexity
            'long': 500     # Complex queries
        }
    
    def classify_query(self, query: str) -> Tuple[QueryComplexity, Dict[str, Any]]:
        """
        Classify a query based on complexity indicators
        
        Returns:
            Tuple of (complexity_level, analysis_details)
        """
        query_lower = query.lower()
        analysis = {
            'length': len(query),
            'word_count': len(query.split()),
            'complexity_score': 0,
            'indicators': [],
            'processing_recommendation': 'standard'
        }
        
        # Length-based scoring
        if analysis['length'] > self.length_thresholds['long']:
            analysis['complexity_score'] += 3
            analysis['indicators'].append('long_query')
        elif analysis['length'] > self.length_thresholds['medium']:
            analysis['complexity_score'] += 2
            analysis['indicators'].append('medium_query')
        else:
            analysis['complexity_score'] += 1
            analysis['indicators'].append('short_query')
        
        # Check for simple indicators
        simple_count = sum(1 for indicator in self.simple_indicators if indicator in query_lower)
        if simple_count > 0:
            analysis['complexity_score'] -= simple_count
            analysis['indicators'].append(f'simple_indicators:{simple_count}')
        
        # Check for complex keywords
        total_complex_score = 0
        for category, keywords in self.complex_keywords.items():
            category_score = sum(1 for keyword in keywords if keyword in query_lower)
            if category_score > 0:
                total_complex_score += category_score
                analysis['indicators'].append(f'{category}:{category_score}')
        
        analysis['complexity_score'] += total_complex_score
        
        # Check for multiple questions or complex structure
        question_marks = query.count('?')
        if question_marks > 1:
            analysis['complexity_score'] += 2
            analysis['indicators'].append(f'multiple_questions:{question_marks}')
        
        # Check for comparison requests
        comparison_words = ['compare', 'versus', 'vs', 'difference between', 'contrast']
        if any(word in query_lower for word in comparison_words):
            analysis['complexity_score'] += 2
            analysis['indicators'].append('comparison_request')
        
        # Check for analytical requests
        analytical_words = ['analyze', 'evaluate', 'assess', 'critique', 'examine']
        if any(word in query_lower for word in analytical_words):
            analysis['complexity_score'] += 2
            analysis['indicators'].append('analytical_request')
        
        # Determine complexity level and processing recommendation
        if analysis['complexity_score'] <= 2:
            complexity = QueryComplexity.SIMPLE
            analysis['processing_recommendation'] = 'standard'  # Use SARVAM + Tavily
        elif analysis['complexity_score'] <= 4:
            complexity = QueryComplexity.MODERATE
            analysis['processing_recommendation'] = 'standard'  # Use SARVAM + Tavily
        elif analysis['complexity_score'] <= 7:
            complexity = QueryComplexity.COMPLEX
            analysis['processing_recommendation'] = 'gpu_assisted'  # Use GPU + SARVAM + Tavily
        else:
            complexity = QueryComplexity.RESEARCH
            analysis['processing_recommendation'] = 'gpu_assisted'  # Use GPU + Advanced processing
        
        logger.info(f"Query classified as {complexity.value} with score {analysis['complexity_score']}")
        return complexity, analysis
    
    def should_use_gpu(self, query: str) -> Tuple[bool, str]:
        """
        Determine if a query should use GPU acceleration
        
        Returns:
            Tuple of (use_gpu: bool, reason: str)
        """
        complexity, analysis = self.classify_query(query)
        
        use_gpu = analysis['processing_recommendation'] == 'gpu_assisted'
        
        if use_gpu:
            reason = f"Complex query ({complexity.value}) requires GPU acceleration"
        else:
            reason = f"Simple/moderate query ({complexity.value}) can use standard APIs"
        
        return use_gpu, reason
    
    def get_processing_strategy(self, query: str) -> Dict[str, Any]:
        """
        Get complete processing strategy for a query
        
        Returns:
            Dictionary with processing recommendations
        """
        complexity, analysis = self.classify_query(query)
        use_gpu, reason = self.should_use_gpu(query)
        
        strategy = {
            'complexity': complexity.value,
            'use_gpu': use_gpu,
            'reason': reason,
            'recommended_apis': [],
            'max_sources': 5,  # Default
            'processing_timeout': 30  # seconds
        }
        
        # Configure APIs based on complexity
        if complexity == QueryComplexity.SIMPLE:
            strategy['recommended_apis'] = ['sarvam']
            strategy['max_sources'] = 3
            strategy['processing_timeout'] = 15
        elif complexity == QueryComplexity.MODERATE:
            strategy['recommended_apis'] = ['sarvam', 'tavily']
            strategy['max_sources'] = 5
            strategy['processing_timeout'] = 30
        elif complexity == QueryComplexity.COMPLEX:
            strategy['recommended_apis'] = ['sarvam', 'tavily', 'gpu_models']
            strategy['max_sources'] = 10
            strategy['processing_timeout'] = 60
        else:  # RESEARCH
            strategy['recommended_apis'] = ['sarvam', 'tavily', 'gpu_models', 'advanced_search']
            strategy['max_sources'] = 15
            strategy['processing_timeout'] = 120
        
        return strategy

# Global classifier instance
query_classifier = QueryClassifier()