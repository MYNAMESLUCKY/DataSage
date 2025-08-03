"""
Analytics Dashboard - Advanced metrics and insights for enterprise RAG system
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import sqlite3
from collections import defaultdict, Counter
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

@dataclass
class QueryMetrics:
    """Metrics for a single query"""
    query_id: str
    timestamp: datetime
    user_id: str
    query_text: str
    processing_time: float
    confidence: float
    model_used: str
    source_count: int
    success: bool
    complexity: str
    processing_strategy: str

@dataclass
class SystemMetrics:
    """Overall system metrics"""
    total_queries: int
    successful_queries: int
    average_processing_time: float
    average_confidence: float
    unique_users: int
    most_used_model: str
    query_complexity_distribution: Dict[str, int]
    processing_strategy_distribution: Dict[str, int]

class AnalyticsDashboard:
    """Advanced analytics dashboard for enterprise RAG system"""
    
    def __init__(self, db_path: str = "analytics.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("analytics_dashboard")
        self._init_database()
    
    def _init_database(self):
        """Initialize analytics database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create analytics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS query_analytics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_id TEXT UNIQUE,
                        timestamp DATETIME,
                        user_id TEXT,
                        query_text TEXT,
                        processing_time REAL,
                        confidence REAL,
                        model_used TEXT,
                        source_count INTEGER,
                        success BOOLEAN,
                        complexity TEXT,
                        processing_strategy TEXT,
                        metadata TEXT
                    )
                """)
                
                # Create user analytics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_analytics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        session_id TEXT,
                        timestamp DATETIME,
                        action_type TEXT,
                        action_data TEXT
                    )
                """)
                
                # Create system metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME,
                        metric_name TEXT,
                        metric_value REAL,
                        metadata TEXT
                    )
                """)
                
                conn.commit()
                self.logger.info("Analytics database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize analytics database: {e}")
    
    def log_query_analytics(self, query_id: str, user_id: str, query_text: str,
                          processing_time: float, success: bool, confidence: float = 0.8,
                          model_used: str = "sarvam-m", source_count: int = 10,
                          complexity: str = "simple", processing_strategy: str = "standard",
                          metadata: Dict[str, Any] = None):
        """Log query analytics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO query_analytics 
                    (query_id, timestamp, user_id, query_text, processing_time, 
                     confidence, model_used, source_count, success, complexity, 
                     processing_strategy, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    query_id,
                    datetime.now(),
                    user_id,
                    query_text[:500],  # Truncate long queries
                    processing_time,
                    confidence,
                    model_used,
                    source_count,
                    success,
                    complexity,
                    processing_strategy,
                    json.dumps(metadata or {})
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to log query analytics: {e}")
    
    def log_user_action(self, user_id: str, session_id: str, action_type: str, action_data: Dict[str, Any] = None):
        """Log user action analytics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO user_analytics 
                    (user_id, session_id, timestamp, action_type, action_data)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    user_id,
                    session_id,
                    datetime.now(),
                    action_type,
                    json.dumps(action_data or {})
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to log user action: {e}")
    
    def get_query_metrics(self, days: int = 30) -> List[QueryMetrics]:
        """Get query metrics for the specified period"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                start_date = datetime.now() - timedelta(days=days)
                
                cursor.execute("""
                    SELECT query_id, timestamp, user_id, query_text, processing_time,
                           confidence, model_used, source_count, success, complexity, processing_strategy
                    FROM query_analytics 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                """, (start_date,))
                
                metrics = []
                for row in cursor.fetchall():
                    metrics.append(QueryMetrics(
                        query_id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        user_id=row[2],
                        query_text=row[3],
                        processing_time=row[4],
                        confidence=row[5],
                        model_used=row[6],
                        source_count=row[7],
                        success=bool(row[8]),
                        complexity=row[9],
                        processing_strategy=row[10]
                    ))
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Failed to get query metrics: {e}")
            return []
    
    def get_system_metrics(self, days: int = 30) -> SystemMetrics:
        """Calculate system-wide metrics"""
        try:
            metrics = self.get_query_metrics(days)
            
            if not metrics:
                return SystemMetrics(
                    total_queries=0,
                    successful_queries=0,
                    average_processing_time=0.0,
                    average_confidence=0.0,
                    unique_users=0,
                    most_used_model="sarvam-m",
                    query_complexity_distribution={},
                    processing_strategy_distribution={}
                )
            
            total_queries = len(metrics)
            successful_queries = sum(1 for m in metrics if m.success)
            average_processing_time = sum(m.processing_time for m in metrics) / total_queries
            average_confidence = sum(m.confidence for m in metrics) / total_queries
            unique_users = len(set(m.user_id for m in metrics))
            
            model_counter = Counter(m.model_used for m in metrics)
            most_used_model = model_counter.most_common(1)[0][0] if model_counter else "sarvam-m"
            
            complexity_distribution = dict(Counter(m.complexity for m in metrics))
            strategy_distribution = dict(Counter(m.processing_strategy for m in metrics))
            
            return SystemMetrics(
                total_queries=total_queries,
                successful_queries=successful_queries,
                average_processing_time=average_processing_time,
                average_confidence=average_confidence,
                unique_users=unique_users,
                most_used_model=most_used_model,
                query_complexity_distribution=complexity_distribution,
                processing_strategy_distribution=strategy_distribution
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate system metrics: {e}")
            return SystemMetrics(0, 0, 0.0, 0.0, 0, "sarvam-m", {}, {})
    
    def generate_performance_chart(self, days: int = 7) -> go.Figure:
        """Generate performance trend chart"""
        try:
            metrics = self.get_query_metrics(days)
            
            if not metrics:
                # Return empty chart
                fig = go.Figure()
                fig.add_annotation(
                    text="No data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
            
            # Group by hour for trend analysis
            df = pd.DataFrame([
                {
                    'timestamp': m.timestamp,
                    'processing_time': m.processing_time,
                    'confidence': m.confidence,
                    'success': m.success
                }
                for m in metrics
            ])
            
            df['hour'] = df['timestamp'].dt.floor('H')
            hourly_stats = df.groupby('hour').agg({
                'processing_time': 'mean',
                'confidence': 'mean',
                'success': 'sum'
            }).reset_index()
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Processing Time (seconds)', 'Confidence Score', 'Successful Queries'),
                vertical_spacing=0.1
            )
            
            # Processing time trend
            fig.add_trace(
                go.Scatter(
                    x=hourly_stats['hour'],
                    y=hourly_stats['processing_time'],
                    mode='lines+markers',
                    name='Processing Time',
                    line=dict(color='#1f77b4', width=2)
                ),
                row=1, col=1
            )
            
            # Confidence trend
            fig.add_trace(
                go.Scatter(
                    x=hourly_stats['hour'],
                    y=hourly_stats['confidence'],
                    mode='lines+markers',
                    name='Confidence',
                    line=dict(color='#ff7f0e', width=2)
                ),
                row=2, col=1
            )
            
            # Success count
            fig.add_trace(
                go.Bar(
                    x=hourly_stats['hour'],
                    y=hourly_stats['success'],
                    name='Successful Queries',
                    marker_color='#2ca02c'
                ),
                row=3, col=1
            )
            
            fig.update_layout(
                height=600,
                title_text="System Performance Trends",
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance chart: {e}")
            return go.Figure()
    
    def generate_complexity_distribution_chart(self, days: int = 30) -> go.Figure:
        """Generate query complexity distribution chart"""
        try:
            system_metrics = self.get_system_metrics(days)
            
            if not system_metrics.query_complexity_distribution:
                fig = go.Figure()
                fig.add_annotation(
                    text="No complexity data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
            
            labels = list(system_metrics.query_complexity_distribution.keys())
            values = list(system_metrics.query_complexity_distribution.values())
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.4,
                    marker_colors=colors[:len(labels)],
                    textinfo='label+percent',
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title="Query Complexity Distribution",
                height=400
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to generate complexity chart: {e}")
            return go.Figure()
    
    def generate_model_usage_chart(self, days: int = 30) -> go.Figure:
        """Generate model usage distribution chart"""
        try:
            metrics = self.get_query_metrics(days)
            
            if not metrics:
                fig = go.Figure()
                fig.add_annotation(
                    text="No model usage data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
            
            model_counts = Counter(m.model_used for m in metrics)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(model_counts.keys()),
                    y=list(model_counts.values()),
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(model_counts)]
                )
            ])
            
            fig.update_layout(
                title="Model Usage Distribution",
                xaxis_title="Model",
                yaxis_title="Query Count",
                height=400
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to generate model usage chart: {e}")
            return go.Figure()
    
    def generate_user_activity_heatmap(self, days: int = 7) -> go.Figure:
        """Generate user activity heatmap"""
        try:
            metrics = self.get_query_metrics(days)
            
            if not metrics:
                fig = go.Figure()
                fig.add_annotation(
                    text="No user activity data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
            
            # Create activity matrix
            df = pd.DataFrame([
                {
                    'day': m.timestamp.strftime('%A'),
                    'hour': m.timestamp.hour,
                    'query_count': 1
                }
                for m in metrics
            ])
            
            activity_matrix = df.groupby(['day', 'hour']).sum().reset_index()
            pivot_table = activity_matrix.pivot(index='day', columns='hour', values='query_count').fillna(0)
            
            # Ensure all days and hours are represented
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            hours_range = list(range(24))
            
            pivot_table = pivot_table.reindex(days_order, fill_value=0)
            pivot_table = pivot_table.reindex(columns=hours_range, fill_value=0)
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot_table.values,
                x=pivot_table.columns,
                y=pivot_table.index,
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Query Count")
            ))
            
            fig.update_layout(
                title="User Activity Heatmap",
                xaxis_title="Hour of Day",
                yaxis_title="Day of Week",
                height=400
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to generate activity heatmap: {e}")
            return go.Figure()
    
    def get_top_queries(self, limit: int = 10, days: int = 30) -> List[Dict[str, Any]]:
        """Get top queries by various metrics"""
        try:
            metrics = self.get_query_metrics(days)
            
            if not metrics:
                return []
            
            # Top by processing time
            slowest_queries = sorted(metrics, key=lambda x: x.processing_time, reverse=True)[:limit]
            
            # Top by confidence
            highest_confidence = sorted(metrics, key=lambda x: x.confidence, reverse=True)[:limit]
            
            # Most common query patterns
            query_patterns = defaultdict(int)
            for m in metrics:
                # Simple pattern extraction (first 3 words)
                pattern = ' '.join(m.query_text.split()[:3]).lower()
                query_patterns[pattern] += 1
            
            top_patterns = sorted(query_patterns.items(), key=lambda x: x[1], reverse=True)[:limit]
            
            return {
                'slowest_queries': [
                    {
                        'query': q.query_text[:100] + '...' if len(q.query_text) > 100 else q.query_text,
                        'processing_time': q.processing_time,
                        'confidence': q.confidence,
                        'timestamp': q.timestamp.isoformat()
                    }
                    for q in slowest_queries
                ],
                'highest_confidence': [
                    {
                        'query': q.query_text[:100] + '...' if len(q.query_text) > 100 else q.query_text,
                        'processing_time': q.processing_time,
                        'confidence': q.confidence,
                        'timestamp': q.timestamp.isoformat()
                    }
                    for q in highest_confidence
                ],
                'common_patterns': [
                    {'pattern': pattern, 'count': count}
                    for pattern, count in top_patterns
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get top queries: {e}")
            return []
    
    def export_analytics_report(self, days: int = 30) -> Dict[str, Any]:
        """Export comprehensive analytics report"""
        try:
            system_metrics = self.get_system_metrics(days)
            top_queries = self.get_top_queries(10, days)
            
            report = {
                'report_generated': datetime.now().isoformat(),
                'period_days': days,
                'system_metrics': asdict(system_metrics),
                'top_queries': top_queries,
                'performance_summary': {
                    'success_rate': (system_metrics.successful_queries / max(system_metrics.total_queries, 1)) * 100,
                    'average_processing_time': system_metrics.average_processing_time,
                    'average_confidence': system_metrics.average_confidence * 100,
                    'total_users': system_metrics.unique_users
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to export analytics report: {e}")
            return {}

# Global analytics instance
analytics_dashboard = AnalyticsDashboard()