"""
RAG Training System for Continuous Improvement
==============================================

This module implements a training system to continuously improve
RAG performance through query analysis, feedback integration,
and model fine-tuning recommendations.
"""

import logging
import json
import time
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import statistics

from .utils import setup_logging

logger = setup_logging(__name__)

@dataclass
class QueryFeedback:
    """Structure for query feedback data"""
    query: str
    answer: str
    sources: List[str]
    confidence: float
    user_satisfied: Optional[bool] = None
    improvement_suggestions: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class PerformanceMetrics:
    """Performance metrics for RAG system"""
    total_queries: int
    avg_confidence: float
    avg_response_time: float
    source_diversity: float
    fallback_usage_rate: float
    user_satisfaction_rate: Optional[float] = None

class RAGTrainingSystem:
    """Training system for continuous RAG improvement"""
    
    def __init__(self):
        self.query_history = []
        self.feedback_data = []
        self.performance_trends = []
        self.improvement_recommendations = []
        self.training_data_file = "training_data.json"
        
        # Load existing training data
        self._load_training_data()
    
    def record_query(
        self, 
        query: str, 
        answer: str, 
        sources: List[str], 
        confidence: float,
        response_time: float,
        fallback_used: bool = False,
        audit_data: Dict = None
    ):
        """Record a query and its results for training analysis"""
        
        query_record = {
            'query': query,
            'answer': answer,
            'sources': sources,
            'confidence': confidence,
            'response_time': response_time,
            'fallback_used': fallback_used,
            'audit_data': audit_data or {},
            'timestamp': time.time(),
            'query_type': self._classify_query_type(query),
            'complexity_score': self._assess_query_complexity(query)
        }
        
        self.query_history.append(query_record)
        
        # Analyze query for immediate insights
        insights = self._analyze_query_performance(query_record)
        if insights:
            logger.info(f"Query insights: {insights}")
        
        # Save training data periodically
        if len(self.query_history) % 10 == 0:
            self._save_training_data()
    
    def add_feedback(self, query: str, user_satisfied: bool, suggestions: str = None):
        """Add user feedback for a specific query"""
        
        # Find the corresponding query record
        matching_query = None
        for record in reversed(self.query_history):
            if record['query'].lower() == query.lower():
                matching_query = record
                break
        
        if matching_query:
            feedback = QueryFeedback(
                query=query,
                answer=matching_query['answer'],
                sources=matching_query['sources'],
                confidence=matching_query['confidence'],
                user_satisfied=user_satisfied,
                improvement_suggestions=suggestions
            )
            
            self.feedback_data.append(feedback)
            logger.info(f"Added feedback for query: {query[:50]}...")
            
            # Generate recommendations based on feedback
            self._analyze_feedback(feedback)
    
    def get_performance_metrics(self, days: int = 7) -> PerformanceMetrics:
        """Get performance metrics for the specified time period"""
        
        cutoff_time = time.time() - (days * 24 * 3600)
        recent_queries = [q for q in self.query_history if q['timestamp'] > cutoff_time]
        
        if not recent_queries:
            return PerformanceMetrics(0, 0.0, 0.0, 0.0, 0.0)
        
        # Calculate metrics
        total_queries = len(recent_queries)
        avg_confidence = statistics.mean(q['confidence'] for q in recent_queries)
        avg_response_time = statistics.mean(q['response_time'] for q in recent_queries)
        
        # Source diversity (unique sources per query)
        source_counts = [len(set(q['sources'])) for q in recent_queries if q['sources']]
        source_diversity = statistics.mean(source_counts) if source_counts else 0.0
        
        # Fallback usage rate
        fallback_queries = sum(1 for q in recent_queries if q.get('fallback_used', False))
        fallback_usage_rate = fallback_queries / total_queries if total_queries > 0 else 0.0
        
        # User satisfaction rate (if feedback available)
        recent_feedback = [f for f in self.feedback_data if f.timestamp > cutoff_time]
        satisfaction_rate = None
        if recent_feedback:
            satisfied_count = sum(1 for f in recent_feedback if f.user_satisfied)
            satisfaction_rate = satisfied_count / len(recent_feedback)
        
        return PerformanceMetrics(
            total_queries=total_queries,
            avg_confidence=avg_confidence,
            avg_response_time=avg_response_time,
            source_diversity=source_diversity,
            fallback_usage_rate=fallback_usage_rate,
            user_satisfaction_rate=satisfaction_rate
        )
    
    def get_improvement_recommendations(self) -> List[Dict[str, Any]]:
        """Get actionable recommendations for system improvement"""
        
        recommendations = []
        
        # Analyze recent performance
        metrics = self.get_performance_metrics(days=30)
        
        # Low confidence queries
        low_confidence_queries = [
            q for q in self.query_history[-100:] 
            if q['confidence'] < 0.6
        ]
        
        if len(low_confidence_queries) > 10:
            recommendations.append({
                'type': 'data_quality',
                'priority': 'high',
                'issue': 'High number of low-confidence responses',
                'suggestion': 'Consider adding more relevant data sources or improving document chunking',
                'affected_queries': len(low_confidence_queries)
            })
        
        # High fallback usage
        if metrics.fallback_usage_rate > 0.3:
            recommendations.append({
                'type': 'coverage',
                'priority': 'medium',
                'issue': f'High fallback usage rate: {metrics.fallback_usage_rate:.1%}',
                'suggestion': 'Expand knowledge base to cover more query types',
                'affected_queries': int(metrics.total_queries * metrics.fallback_usage_rate)
            })
        
        # Slow response times
        if metrics.avg_response_time > 10.0:
            recommendations.append({
                'type': 'performance',
                'priority': 'medium', 
                'issue': f'Slow average response time: {metrics.avg_response_time:.1f}s',
                'suggestion': 'Optimize vector search or reduce document chunk sizes',
                'impact': 'user_experience'
            })
        
        # Query type analysis
        query_type_analysis = self._analyze_query_types()
        for analysis in query_type_analysis:
            if analysis['needs_improvement']:
                recommendations.append(analysis)
        
        # User feedback analysis
        if metrics.user_satisfaction_rate and metrics.user_satisfaction_rate < 0.7:
            recommendations.append({
                'type': 'satisfaction',
                'priority': 'high',
                'issue': f'Low user satisfaction: {metrics.user_satisfaction_rate:.1%}',
                'suggestion': 'Review user feedback and improve answer quality',
                'user_feedback_count': len(self.feedback_data)
            })
        
        return sorted(recommendations, key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['priority']], reverse=True)
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query"""
        query_lower = query.lower()
        
        if any(pattern in query_lower for pattern in ['what is', 'what are', 'define']):
            return 'definition'
        elif any(pattern in query_lower for pattern in ['how to', 'how do', 'how can']):
            return 'how_to'
        elif any(pattern in query_lower for pattern in ['why', 'why does', 'why is']):
            return 'explanation'
        elif any(pattern in query_lower for pattern in ['when', 'when did', 'when was']):
            return 'temporal'
        elif any(pattern in query_lower for pattern in ['who', 'who is', 'who was']):
            return 'person'
        elif any(pattern in query_lower for pattern in ['where', 'where is', 'where was']):
            return 'location'
        elif any(pattern in query_lower for pattern in ['compare', 'difference', 'vs']):
            return 'comparison'
        elif any(pattern in query_lower for pattern in ['list', 'examples', 'types of']):
            return 'list'
        else:
            return 'general'
    
    def _assess_query_complexity(self, query: str) -> float:
        """Assess the complexity of a query (0-1 scale)"""
        complexity = 0.3  # Base complexity
        
        # Length factor
        word_count = len(query.split())
        complexity += min(word_count / 20, 0.3)
        
        # Multiple concepts
        concept_indicators = ['and', 'or', 'but', 'however', 'also', 'additionally']
        complexity += sum(0.1 for indicator in concept_indicators if indicator in query.lower())
        
        # Technical terms (simplified check)
        technical_indicators = ['algorithm', 'methodology', 'implementation', 'analysis']
        complexity += sum(0.1 for term in technical_indicators if term in query.lower())
        
        return min(complexity, 1.0)
    
    def _analyze_query_performance(self, query_record: Dict) -> Optional[str]:
        """Analyze individual query performance for immediate insights"""
        insights = []
        
        # Low confidence warning
        if query_record['confidence'] < 0.5:
            insights.append("Low confidence response - consider data source expansion")
        
        # Slow response warning
        if query_record['response_time'] > 15.0:
            insights.append("Slow response time - consider optimization")
        
        # Fallback usage
        if query_record.get('fallback_used', False):
            insights.append("Fallback search used - knowledge base gap identified")
        
        return "; ".join(insights) if insights else None
    
    def _analyze_feedback(self, feedback: QueryFeedback):
        """Analyze user feedback for improvement opportunities"""
        if not feedback.user_satisfied:
            # Log negative feedback for analysis
            logger.warning(f"Negative feedback for query: {feedback.query[:50]}...")
            
            if feedback.improvement_suggestions:
                logger.info(f"User suggestions: {feedback.improvement_suggestions}")
            
            # Add to improvement recommendations
            self.improvement_recommendations.append({
                'type': 'user_feedback',
                'priority': 'high',
                'query': feedback.query,
                'issue': 'User dissatisfaction',
                'user_suggestions': feedback.improvement_suggestions,
                'timestamp': feedback.timestamp
            })
    
    def _analyze_query_types(self) -> List[Dict[str, Any]]:
        """Analyze performance by query type"""
        recommendations = []
        
        # Group queries by type
        type_performance = defaultdict(list)
        for query in self.query_history[-200:]:  # Last 200 queries
            query_type = query['query_type']
            type_performance[query_type].append(query)
        
        # Analyze each type
        for query_type, queries in type_performance.items():
            if len(queries) < 5:  # Skip types with too few samples
                continue
            
            avg_confidence = statistics.mean(q['confidence'] for q in queries)
            fallback_rate = sum(1 for q in queries if q.get('fallback_used', False)) / len(queries)
            
            needs_improvement = False
            issues = []
            
            if avg_confidence < 0.6:
                issues.append(f"low confidence ({avg_confidence:.2f})")
                needs_improvement = True
            
            if fallback_rate > 0.4:
                issues.append(f"high fallback usage ({fallback_rate:.1%})")
                needs_improvement = True
            
            if needs_improvement:
                recommendations.append({
                    'type': 'query_type_performance',
                    'priority': 'medium',
                    'query_type': query_type,
                    'issue': f"Poor performance for {query_type} queries: {', '.join(issues)}",
                    'suggestion': f"Focus on improving data sources for {query_type} questions",
                    'sample_count': len(queries),
                    'needs_improvement': True
                })
            else:
                recommendations.append({
                    'query_type': query_type,
                    'needs_improvement': False
                })
        
        return recommendations
    
    def _load_training_data(self):
        """Load existing training data from disk"""
        try:
            if os.path.exists(self.training_data_file):
                with open(self.training_data_file, 'r') as f:
                    data = json.load(f)
                    self.query_history = data.get('query_history', [])
                    
                    # Convert feedback data
                    feedback_list = data.get('feedback_data', [])
                    self.feedback_data = [
                        QueryFeedback(**fb) for fb in feedback_list
                    ]
                    
                logger.info(f"Loaded {len(self.query_history)} query records and {len(self.feedback_data)} feedback entries")
        except Exception as e:
            logger.warning(f"Failed to load training data: {str(e)}")
    
    def _save_training_data(self):
        """Save training data to disk"""
        try:
            data = {
                'query_history': self.query_history,
                'feedback_data': [asdict(fb) for fb in self.feedback_data],
                'last_updated': time.time()
            }
            
            with open(self.training_data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug("Saved training data")
        except Exception as e:
            logger.warning(f"Failed to save training data: {str(e)}")
    
    def export_training_insights(self) -> Dict[str, Any]:
        """Export comprehensive training insights for external analysis"""
        metrics = self.get_performance_metrics(days=30)
        recommendations = self.get_improvement_recommendations()
        
        # Query type distribution
        query_types = Counter(q['query_type'] for q in self.query_history[-200:])
        
        # Common issues
        low_confidence_queries = [
            {'query': q['query'], 'confidence': q['confidence'], 'sources': len(q['sources'])}
            for q in self.query_history[-100:]
            if q['confidence'] < 0.6
        ]
        
        return {
            'performance_metrics': asdict(metrics),
            'recommendations': recommendations,
            'query_type_distribution': dict(query_types),
            'low_confidence_examples': low_confidence_queries[:10],
            'total_queries_analyzed': len(self.query_history),
            'feedback_entries': len(self.feedback_data),
            'generated_at': time.time()
        }

# Global instance
_training_system = None

def get_training_system() -> RAGTrainingSystem:
    """Get global training system instance"""
    global _training_system
    if _training_system is None:
        _training_system = RAGTrainingSystem()
    return _training_system