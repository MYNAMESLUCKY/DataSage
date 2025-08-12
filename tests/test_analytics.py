"""
Test suite for analytics and feedback components
"""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import sqlite3
from src.backend.analytics_dashboard import AnalyticsDashboard

class TestAnalytics(unittest.TestCase):
    def setUp(self):
        # Create a new analytics dashboard with in-memory database for testing
        self.analytics = AnalyticsDashboard(db_path=":memory:")
        # Initialize database explicitly for testing
        self.analytics._init_database()
        
    def test_query_metrics(self):
        """Test if query metrics are properly logged"""
        query_id = "test_1"
        user_id = "test_user"
        query_text = "Test query"
        processing_time = 1.5
        
        # Log a test query
        self.analytics.log_query_analytics(
            query_id=query_id,
            user_id=user_id,
            query_text=query_text,
            processing_time=processing_time,
            confidence=0.95,
            model_used="test_model",
            source_count=3,
            success=True,
            complexity="SIMPLE",
            processing_strategy="standard"
        )
        
        # Get and verify metrics
        metrics = self.analytics.get_query_metrics()
        self.assertGreater(len(metrics), 0)
        
        # Verify the first metric
        first_metric = metrics[0]
        self.assertEqual(first_metric.query_id, query_id)
        self.assertEqual(first_metric.user_id, user_id)
        self.assertEqual(first_metric.query_text, query_text)
        self.assertEqual(first_metric.processing_time, processing_time)
        
    def test_user_actions(self):
        """Test user action tracking"""
        user_id = "test_user"
        session_id = "test_session"
        action_type = "search"
        action_data = {"query": "test query"}
        
        self.analytics.log_user_action(
            user_id=user_id,
            session_id=session_id,
            action_type=action_type,
            action_data=action_data
        )
        
        # Verify the action was logged
        with sqlite3.connect(self.analytics.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM user_analytics WHERE user_id = ?", (user_id,))
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)

class TestAnalyticsDashboardHealth(unittest.TestCase):
    def test_healthy_in_memory(self):
        # Use in-memory database to test healthy condition
        dashboard = AnalyticsDashboard(db_path=':memory:')
        health = dashboard.check_database_health()
        self.assertEqual(health['status'], 'healthy')
        self.assertEqual(health['integrity'], 'ok')
        for table in ['query_analytics', 'user_analytics', 'system_metrics']:
            self.assertIn(table, health['tables'])
            self.assertIsInstance(health['tables'][table]['row_count'], int)
            self.assertIsInstance(health['tables'][table]['index_count'], int)

    def test_missing_db_file(self):
        # Create a temporary file-based database, then remove the file to simulate missing database
        temp_db = 'temp_test_missing.db'
        if os.path.exists(temp_db):
            os.remove(temp_db)
        # Creating instance will initialize and create the DB file
        dashboard = AnalyticsDashboard(db_path=temp_db)
        # Remove the file manually to simulate missing database file
        os.remove(temp_db)
        health = dashboard.check_database_health()
        self.assertEqual(health['status'], 'unhealthy')
        self.assertTrue(any('Database file not found' in err for err in health['errors']))

class TestAPIKeyCheck(unittest.TestCase):
    """Test API key validation functionality"""
    
    def setUp(self):
        self.dashboard = AnalyticsDashboard(db_path=':memory:')
        # Store original environment variables
        self.original_tavily_key = os.getenv('TAVILY_API_KEY')
        self.original_sarvam_key = os.getenv('SARVAM_API_KEY')

    def tearDown(self):
        # Restore original environment variables
        if self.original_tavily_key:
            os.environ['TAVILY_API_KEY'] = self.original_tavily_key
        elif 'TAVILY_API_KEY' in os.environ:
            del os.environ['TAVILY_API_KEY']
            
        if self.original_sarvam_key:
            os.environ['SARVAM_API_KEY'] = self.original_sarvam_key
        elif 'SARVAM_API_KEY' in os.environ:
            del os.environ['SARVAM_API_KEY']

    def test_missing_api_keys(self):
        # Remove API keys from environment
        if 'TAVILY_API_KEY' in os.environ:
            del os.environ['TAVILY_API_KEY']
        if 'SARVAM_API_KEY' in os.environ:
            del os.environ['SARVAM_API_KEY']
            
        status = self.dashboard.check_api_keys()
        self.assertEqual(status['status'], 'unhealthy')
        self.assertEqual(status['apis']['tavily']['status'], 'error')
        self.assertEqual(status['apis']['sarvam']['status'], 'error')
        self.assertIn('not found in environment', status['apis']['tavily']['error'])
        self.assertIn('not found in environment', status['apis']['sarvam']['error'])

    def test_api_keys_present(self):
        if not os.getenv('TAVILY_API_KEY') or not os.getenv('SARVAM_API_KEY'):
            self.skipTest("Skipping API test because keys are not set")
            
        status = self.dashboard.check_api_keys()
        self.assertEqual(status['status'], 'healthy')
        for api in ['tavily', 'sarvam']:
            self.assertEqual(status['apis'][api]['status'], 'healthy')
            self.assertIsNone(status['apis'][api]['error'])


if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    unittest.main()
