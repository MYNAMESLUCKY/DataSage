"""
Complexity Classification System for Query Processing

This module classifies query complexity on a scale of 0-1 to determine
if GPU processing is required for heavy computational tasks.
"""

import re
import logging
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ComplexityLevel(Enum):
    SIMPLE = "simple"          # 0.0 - 0.3
    MODERATE = "moderate"      # 0.3 - 0.6  
    COMPLEX = "complex"        # 0.6 - 0.8
    VERY_COMPLEX = "very_complex"  # 0.8 - 1.0

@dataclass
class ComplexityAnalysis:
    score: float  # 0.0 to 1.0
    level: ComplexityLevel
    factors: Dict[str, float]
    reasoning: str
    gpu_recommended: bool
    estimated_compute_time: float  # seconds

class QueryComplexityClassifier:
    """Advanced classifier for query complexity analysis"""
    
    def __init__(self):
        self.complexity_patterns = {
            # Mathematical/Scientific complexity indicators
            'mathematical': {
                'patterns': [
                    r'\b(equation|formula|calculate|compute|algorithm|optimization)\b',
                    r'\b(integral|derivative|matrix|vector|tensor|probability)\b',
                    r'\b(simulation|modeling|statistical|regression|neural)\b',
                    r'\b(quantum|physics|thermodynamics|electromagnetic)\b'
                ],
                'weight': 0.25
            },
            
            # Multi-step reasoning complexity
            'reasoning': {
                'patterns': [
                    r'\b(analyze|compare|evaluate|synthesize|correlate)\b',
                    r'\b(relationship|causation|implication|consequence)\b',
                    r'\b(hypothesis|theory|framework|methodology)\b',
                    r'\b(comprehensive|thorough|detailed|extensive)\b'
                ],
                'weight': 0.20
            },
            
            # Data processing complexity
            'data_processing': {
                'patterns': [
                    r'\b(dataset|database|big data|processing|transformation)\b',
                    r'\b(aggregation|filtering|sorting|clustering|classification)\b',
                    r'\b(machine learning|deep learning|AI|neural network)\b',
                    r'\b(pattern recognition|feature extraction|dimensionality)\b'
                ],
                'weight': 0.25
            },
            
            # Philosophical/Abstract complexity
            'abstract': {
                'patterns': [
                    r'\b(philosophical|theoretical|conceptual|metaphysical)\b',
                    r'\b(consciousness|existence|reality|truth|meaning)\b',
                    r'\b(ethical|moral|values|principles|beliefs)\b',
                    r'\b(paradox|dilemma|contradiction|ambiguity)\b'
                ],
                'weight': 0.15
            },
            
            # Multi-domain complexity
            'interdisciplinary': {
                'patterns': [
                    r'\b(interdisciplinary|multidisciplinary|cross-domain)\b',
                    r'\b(integration|synthesis|combination|merger)\b',
                    r'\b(holistic|comprehensive|systematic|interconnected)\b'
                ],
                'weight': 0.15
            }
        }
        
        # Question structure complexity indicators
        self.structure_indicators = {
            'multiple_questions': r'\?.*\?',  # Multiple question marks
            'conditional_logic': r'\b(if|when|unless|provided|given that|assuming)\b',
            'comparisons': r'\b(versus|compared to|in contrast|differences|similarities)\b',
            'temporal_reasoning': r'\b(before|after|during|while|timeline|sequence)\b',
            'causal_chains': r'\b(because|therefore|thus|consequently|leads to|results in)\b'
        }
        
        logger.info("Query complexity classifier initialized")
    
    def classify_complexity(self, query: str) -> ComplexityAnalysis:
        """
        Classify query complexity on a scale of 0-1
        
        Args:
            query: The input query string
            
        Returns:
            ComplexityAnalysis with score, level, and reasoning
        """
        query_lower = query.lower()
        factors = {}
        total_score = 0.0
        
        # 1. Pattern-based complexity scoring
        for category, config in self.complexity_patterns.items():
            category_score = 0.0
            matches = 0
            
            for pattern in config['patterns']:
                pattern_matches = len(re.findall(pattern, query_lower))
                if pattern_matches > 0:
                    matches += pattern_matches
            
            if matches > 0:
                # Normalize by query length and apply diminishing returns
                normalized_score = min(1.0, matches / (len(query.split()) / 10))
                category_score = normalized_score * config['weight']
                
            factors[f'{category}_patterns'] = category_score
            total_score += category_score
        
        # 2. Structural complexity analysis
        structure_score = 0.0
        
        # Multiple questions increase complexity
        question_marks = query.count('?')
        if question_marks > 1:
            structure_score += min(0.1, question_marks * 0.03)
            factors['multiple_questions'] = min(0.1, question_marks * 0.03)
        
        # Sentence structure complexity
        sentences = query.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        if avg_sentence_length > 20:
            length_complexity = min(0.15, (avg_sentence_length - 20) * 0.01)
            structure_score += length_complexity
            factors['sentence_complexity'] = length_complexity
        
        # Conditional and logical structure
        for indicator, pattern in self.structure_indicators.items():
            matches = len(re.findall(pattern, query_lower))
            if matches > 0:
                indicator_score = min(0.05, matches * 0.02)
                structure_score += indicator_score
                factors[f'structure_{indicator}'] = indicator_score
        
        total_score += structure_score
        
        # 3. Vocabulary complexity
        vocab_score = self._analyze_vocabulary_complexity(query)
        factors['vocabulary_complexity'] = vocab_score
        total_score += vocab_score
        
        # 4. Domain-specific complexity boosters
        domain_score = self._analyze_domain_complexity(query_lower)
        factors['domain_complexity'] = domain_score
        total_score += domain_score
        
        # Normalize final score to 0-1 range
        final_score = min(1.0, total_score)
        
        # Determine complexity level and GPU recommendation
        if final_score < 0.3:
            level = ComplexityLevel.SIMPLE
            gpu_recommended = False
            estimated_time = 1.0
        elif final_score < 0.6:
            level = ComplexityLevel.MODERATE
            gpu_recommended = False
            estimated_time = 3.0
        elif final_score < 0.8:
            level = ComplexityLevel.COMPLEX
            gpu_recommended = True
            estimated_time = 8.0
        else:
            level = ComplexityLevel.VERY_COMPLEX
            gpu_recommended = True
            estimated_time = 15.0
        
        # Generate reasoning
        reasoning = self._generate_reasoning(factors, final_score, level)
        
        logger.info(f"Query complexity: {final_score:.3f} ({level.value}) - GPU: {gpu_recommended}")
        
        return ComplexityAnalysis(
            score=final_score,
            level=level,
            factors=factors,
            reasoning=reasoning,
            gpu_recommended=gpu_recommended,
            estimated_compute_time=estimated_time
        )
    
    def _analyze_vocabulary_complexity(self, query: str) -> float:
        """Analyze vocabulary complexity based on word characteristics"""
        words = query.lower().split()
        if not words:
            return 0.0
        
        complexity_score = 0.0
        
        # Long words (technical terminology)
        long_words = [w for w in words if len(w) > 12]
        if long_words:
            complexity_score += min(0.1, len(long_words) / len(words))
        
        # Academic/technical vocabulary
        technical_prefixes = ['meta', 'trans', 'inter', 'multi', 'pseudo', 'proto']
        technical_suffixes = ['tion', 'sion', 'ment', 'ness', 'ity', 'ism', 'ogy']
        
        technical_words = 0
        for word in words:
            if (any(word.startswith(prefix) for prefix in technical_prefixes) or
                any(word.endswith(suffix) for suffix in technical_suffixes)):
                technical_words += 1
        
        if technical_words > 0:
            complexity_score += min(0.1, technical_words / len(words))
        
        return complexity_score
    
    def _analyze_domain_complexity(self, query: str) -> float:
        """Analyze domain-specific complexity indicators"""
        
        # High-complexity domains
        complex_domains = {
            'quantum_physics': ['quantum', 'entanglement', 'superposition', 'uncertainty'],
            'neuroscience': ['neural', 'synaptic', 'neurotransmitter', 'consciousness'],
            'philosophy': ['ontological', 'epistemological', 'phenomenological', 'existential'],
            'mathematics': ['topology', 'manifold', 'differential', 'stochastic'],
            'ai_ml': ['transformer', 'gradient', 'backpropagation', 'reinforcement'],
            'economics': ['macroeconomic', 'microeconomic', 'econometric', 'behavioral']
        }
        
        domain_score = 0.0
        for domain, keywords in complex_domains.items():
            matches = sum(1 for keyword in keywords if keyword in query)
            if matches > 0:
                domain_score += min(0.05, matches * 0.02)
        
        return domain_score
    
    def _generate_reasoning(self, factors: Dict[str, float], score: float, level: ComplexityLevel) -> str:
        """Generate human-readable reasoning for complexity classification"""
        
        reasoning_parts = [f"Query classified as {level.value.upper()} (score: {score:.3f})"]
        
        # Identify top contributing factors
        sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
        significant_factors = [(k, v) for k, v in sorted_factors if v > 0.05]
        
        if significant_factors:
            reasoning_parts.append("Key complexity indicators:")
            for factor, value in significant_factors[:3]:
                reasoning_parts.append(f"  • {factor.replace('_', ' ').title()}: {value:.3f}")
        
        if level in [ComplexityLevel.COMPLEX, ComplexityLevel.VERY_COMPLEX]:
            reasoning_parts.append("GPU processing recommended for optimal performance")
        
        return "\n".join(reasoning_parts)

# Global classifier instance
complexity_classifier = QueryComplexityClassifier()

def classify_query_complexity(query: str) -> ComplexityAnalysis:
    """Convenience function for complexity classification"""
    return complexity_classifier.classify_complexity(query)