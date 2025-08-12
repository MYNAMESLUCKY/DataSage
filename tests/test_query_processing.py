"""
Test suite for query processing and RAG system components
"""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backend.hybrid_rag_processor import HybridRAGProcessor
from src.backend.agentic_rag import AgentRole
from src.backend.vector_store_chroma import ChromaVectorStoreManager
from src.backend.tavily_integration import TavilyIntegrationService

class TestQueryProcessing(unittest.TestCase):
    def setUp(self):
        self.mock_db = MagicMock(spec=ChromaVectorStoreManager)
        self.mock_tavily = MagicMock(spec=TavilyIntegrationService)
        
    def test_basic_query_processing(self):
        """Test basic query processing without web search"""
        query = "What is machine learning?"
        processor = HybridRAGProcessor()
        response = processor.process_query(query)
        self.assertIsNotNone(response)
        self.assertTrue(isinstance(response, dict))
        self.assertIn('answer', response)
        
    @patch('src.backend.tavily_integration.TavilySearch.search')
    def test_web_enhanced_query(self, mock_search):
        """Test query processing with web search integration"""
        mock_search.return_value = {"results": [{"content": "Test content"}]}
        query = "What are the latest developments in AI?"
        processor = HybridRAGProcessor()
        response = processor.process_query(query, use_web_search=True)
        self.assertIn('web_results', response)
        
    def test_vector_store_retrieval(self):
        """Test vector store document retrieval"""
        self.mock_db.query.return_value = [{"content": "Test document"}]
        query = "Python programming"
        results = self.mock_db.query(query)
        self.assertIsNotNone(results)
        self.assertTrue(len(results) > 0)

if __name__ == '__main__':
    unittest.main()
