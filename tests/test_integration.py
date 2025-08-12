"""
Integration tests for the entire query processing pipeline
"""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backend.hybrid_rag_processor import HybridRAGProcessor
from src.backend.analytics_dashboard import AnalyticsDashboard
from src.backend.agentic_rag import AgentRole
from src.backend.tavily_integration import TavilyIntegrationService

class TestIntegration(unittest.TestCase):
    def setUp(self):
        # Mock required components
        self.mock_vector_store = MagicMock()
        self.mock_rag_engine = MagicMock()
        self.mock_enhanced_retrieval = MagicMock()
        
        self.rag_processor = HybridRAGProcessor(
            vector_store=self.mock_vector_store,
            rag_engine=self.mock_rag_engine,
            enhanced_retrieval=self.mock_enhanced_retrieval
        )
        self.analytics = AnalyticsDashboard()
        self.mock_tavily = MagicMock(spec=TavilyIntegrationService)
        
    def test_end_to_end_query(self):
        """Test complete query processing pipeline"""
        query = "Explain the concept of quantum computing"
        
        # Process query
        response = self.rag_processor.process_intelligent_query(query)
        self.assertIsNotNone(response)
        
        # Check analytics
        metrics = self.analytics.get_query_metrics()
        self.assertIsInstance(metrics, list)
        
        # Verify response structure
        self.assertIn('answer', response)
        self.assertIn('sources', response)
        
    @patch('src.backend.tavily_integration.TavilyIntegrationService.search_and_fetch')
    def test_web_enhanced_query(self, mock_search):
        """Test query processing with web search"""
        query = "What are the latest breakthroughs in fusion energy research?"
        
        mock_search.return_value = {
            "results": [{"content": "Recent fusion breakthrough at ITER"}]
        }
        
        response = self.rag_processor.process_intelligent_query(query, use_web_search=True)
        self.assertIn('web_sources', response)
            
    def test_error_handling(self):
        """Test error handling in the pipeline"""
        query = "Test error handling"
        
        # Simulate ChromaDB error
        with patch('src.backend.vector_store_chroma.ChromaVectorStoreManager.similarity_search',
                  side_effect=Exception("DB Error")):
            response = self.rag_processor.process_intelligent_query(query)
            self.assertEqual(response['status'], 'error')
            
            # Check error tracking
            error_stats = self.analytics.get_error_stats()
            self.assertIn('total_errors', error_stats)

if __name__ == '__main__':
    unittest.main()
