"""
Enterprise Analytics Dashboard
Real-time analytics and monitoring dashboard using Streamlit
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List

from src.analytics.system_monitor import system_monitor, QueryMetrics
import logging

logger = logging.getLogger(__name__)

class AnalyticsDashboard:
    """Enterprise analytics dashboard with real-time monitoring"""
    
    def __init__(self):
        self.system_monitor = system_monitor
        
    def render_dashboard(self):
        """Render the complete analytics dashboard"""
        st.set_page_config(
            page_title="Enterprise RAG Analytics",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for professional styling
        st.markdown("""
        <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 0.5rem;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
        }
        .alert-critical {
            background-color: #ff6b6b;
            padding: 0.5rem;
            border-radius: 0.25rem;
            color: white;
            margin: 0.25rem 0;
        }
        .alert-warning {
            background-color: #ffa726;
            padding: 0.5rem;
            border-radius: 0.25rem;
            color: white;
            margin: 0.25rem 0;
        }
        .status-good {
            color: #4caf50;
            font-weight: bold;
        }
        .status-warning {
            color: #ff9800;
            font-weight: bold;
        }
        .status-critical {
            color: #f44336;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.title("🏢 Enterprise RAG Analytics Dashboard")
        st.markdown("---")
        
        # Sidebar controls
        self._render_sidebar()
        
        # Main dashboard content
        self._render_overview()
        self._render_system_monitoring()
        self._render_query_analytics()
        self._render_cost_tracking()
        self._render_performance_alerts()
        
    def _render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("⚙️ Dashboard Controls")
        
        # Time range selection
        time_ranges = {
            "Last Hour": 1,
            "Last 6 Hours": 6,
            "Last 24 Hours": 24,
            "Last 7 Days": 168,
            "Last 30 Days": 720
        }
        
        selected_range = st.sidebar.selectbox(
            "📅 Time Range",
            options=list(time_ranges.keys()),
            index=2  # Default to 24 hours
        )
        
        st.session_state.time_range_hours = time_ranges[selected_range]
        
        # Auto-refresh controls
        st.sidebar.subheader("🔄 Auto Refresh")
        auto_refresh = st.sidebar.checkbox("Enable Auto Refresh", value=True)
        
        if auto_refresh:
            refresh_interval = st.sidebar.slider(
                "Refresh Interval (seconds)",
                min_value=10,
                max_value=300,
                value=30,
                step=10
            )
            
            # Auto refresh using rerun
            if auto_refresh:
                time.sleep(refresh_interval)
                st.rerun()
        
        # System controls
        st.sidebar.subheader("🖥️ System Controls")
        
        if st.sidebar.button("🚀 Start Monitoring"):
            self.system_monitor.start_monitoring(interval=30)
            st.sidebar.success("Monitoring started!")
            
        if st.sidebar.button("⏹️ Stop Monitoring"):
            self.system_monitor.stop_monitoring()
            st.sidebar.success("Monitoring stopped!")
            
        if st.sidebar.button("🧹 Cleanup Old Data"):
            self.system_monitor.cleanup_old_data(days=30)
            st.sidebar.success("Old data cleaned up!")
            
    def _render_overview(self):
        """Render system overview metrics"""
        st.header("📊 System Overview")
        
        # Get current system metrics
        current_metrics = self.system_monitor._collect_system_metrics()
        query_analytics = self.system_monitor.get_query_analytics(
            hours=st.session_state.get('time_range_hours', 24)
        )
        cost_analytics = self.system_monitor.get_cost_analytics(
            hours=st.session_state.get('time_range_hours', 24)
        )
        
        # Create metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cpu_status = "🟢" if current_metrics.cpu_percent < 70 else "🟡" if current_metrics.cpu_percent < 90 else "🔴"
            st.metric(
                label=f"{cpu_status} CPU Usage",
                value=f"{current_metrics.cpu_percent:.1f}%",
                delta=None
            )
            
        with col2:
            mem_status = "🟢" if current_metrics.memory_percent < 70 else "🟡" if current_metrics.memory_percent < 90 else "🔴"
            st.metric(
                label=f"{mem_status} Memory Usage",
                value=f"{current_metrics.memory_percent:.1f}%",
                delta=f"{current_metrics.memory_used_mb:.0f} MB used"
            )
            
        with col3:
            st.metric(
                label="📈 Total Queries",
                value=f"{query_analytics.get('total_queries', 0):,}",
                delta=f"{query_analytics.get('success_rate', 0):.1f}% success rate"
            )
            
        with col4:
            st.metric(
                label="💰 Total Cost",
                value=f"${cost_analytics.get('total_cost', 0):.4f}",
                delta=f"{cost_analytics.get('total_tokens', 0):,} tokens"
            )
            
    def _render_system_monitoring(self):
        """Render real-time system monitoring charts"""
        st.header("🖥️ System Monitoring")
        
        # Get system metrics
        time_range = st.session_state.get('time_range_hours', 24)
        metrics_data = self.system_monitor.get_system_metrics(hours=time_range)
        
        if not metrics_data:
            st.warning("No system metrics available. Start monitoring to collect data.")
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(metrics_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create subplots for system metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU Usage (%)', 'Memory Usage (%)', 'Disk Usage (%)', 'Active Connections'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # CPU Usage
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['cpu_percent'],
                      mode='lines+markers', name='CPU %',
                      line=dict(color='#1f77b4', width=2)),
            row=1, col=1
        )
        
        # Memory Usage
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['memory_percent'],
                      mode='lines+markers', name='Memory %',
                      line=dict(color='#ff7f0e', width=2)),
            row=1, col=2
        )
        
        # Disk Usage
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['disk_usage_percent'],
                      mode='lines+markers', name='Disk %',
                      line=dict(color='#2ca02c', width=2)),
            row=2, col=1
        )
        
        # Active Connections
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['active_connections'],
                      mode='lines+markers', name='Connections',
                      line=dict(color='#d62728', width=2)),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Real-time System Metrics"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # System status summary
        col1, col2, col3 = st.columns(3)
        
        current_metrics = self.system_monitor._collect_system_metrics()
        
        with col1:
            if current_metrics.cpu_percent < 70:
                st.markdown('<p class="status-good">✅ CPU Status: Normal</p>', unsafe_allow_html=True)
            elif current_metrics.cpu_percent < 90:
                st.markdown('<p class="status-warning">⚠️ CPU Status: High Usage</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-critical">🚨 CPU Status: Critical</p>', unsafe_allow_html=True)
                
        with col2:
            if current_metrics.memory_percent < 70:
                st.markdown('<p class="status-good">✅ Memory Status: Normal</p>', unsafe_allow_html=True)
            elif current_metrics.memory_percent < 90:
                st.markdown('<p class="status-warning">⚠️ Memory Status: High Usage</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-critical">🚨 Memory Status: Critical</p>', unsafe_allow_html=True)
                
        with col3:
            if current_metrics.disk_usage_percent < 80:
                st.markdown('<p class="status-good">✅ Storage Status: Normal</p>', unsafe_allow_html=True)
            elif current_metrics.disk_usage_percent < 95:
                st.markdown('<p class="status-warning">⚠️ Storage Status: High Usage</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-critical">🚨 Storage Status: Critical</p>', unsafe_allow_html=True)
                
    def _render_query_analytics(self):
        """Render query performance analytics"""
        st.header("🔍 Query Analytics")
        
        time_range = st.session_state.get('time_range_hours', 24)
        analytics = self.system_monitor.get_query_analytics(hours=time_range)
        
        if not analytics or analytics.get('total_queries', 0) == 0:
            st.info("No query data available for the selected time range.")
            return
            
        # Query performance overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Queries",
                value=f"{analytics['total_queries']:,}"
            )
            
        with col2:
            st.metric(
                label="Avg Response Time",
                value=f"{analytics['avg_processing_time']:.2f}s"
            )
            
        with col3:
            st.metric(
                label="Success Rate",
                value=f"{analytics['success_rate']:.1f}%"
            )
            
        with col4:
            st.metric(
                label="Avg Quality Score",
                value=f"{analytics['avg_quality_score']:.2f}/5.0"
            )
            
        # Query patterns by hour
        if analytics.get('hourly_patterns'):
            hourly_df = pd.DataFrame(
                analytics['hourly_patterns'],
                columns=['hour', 'query_count', 'avg_response_time']
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_queries = px.bar(
                    hourly_df,
                    x='hour',
                    y='query_count',
                    title='Queries by Hour',
                    labels={'hour': 'Hour of Day', 'query_count': 'Number of Queries'}
                )
                st.plotly_chart(fig_queries, use_container_width=True)
                
            with col2:
                fig_response = px.line(
                    hourly_df,
                    x='hour',
                    y='avg_response_time',
                    title='Average Response Time by Hour',
                    labels={'hour': 'Hour of Day', 'avg_response_time': 'Response Time (s)'}
                )
                st.plotly_chart(fig_response, use_container_width=True)
                
        # Model usage statistics
        if analytics.get('model_usage'):
            model_df = pd.DataFrame(
                analytics['model_usage'],
                columns=['model', 'query_count', 'avg_response_time', 'avg_cost']
            )
            
            st.subheader("📊 Model Usage Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_model_usage = px.pie(
                    model_df,
                    values='query_count',
                    names='model',
                    title='Query Distribution by Model'
                )
                st.plotly_chart(fig_model_usage, use_container_width=True)
                
            with col2:
                fig_model_performance = px.bar(
                    model_df,
                    x='model',
                    y='avg_response_time',
                    title='Average Response Time by Model',
                    labels={'avg_response_time': 'Response Time (s)'}
                )
                st.plotly_chart(fig_model_performance, use_container_width=True)
                
        # Top users (if available)
        if analytics.get('top_users'):
            st.subheader("👥 Top Users")
            users_df = pd.DataFrame(
                analytics['top_users'],
                columns=['user_id', 'query_count', 'avg_response_time']
            )
            st.dataframe(users_df, use_container_width=True)
            
    def _render_cost_tracking(self):
        """Render cost tracking and analytics"""
        st.header("💰 Cost Tracking")
        
        time_range = st.session_state.get('time_range_hours', 24)
        cost_data = self.system_monitor.get_cost_analytics(hours=time_range)
        
        if not cost_data or cost_data.get('total_cost', 0) == 0:
            st.info("No cost data available for the selected time range.")
            return
            
        # Cost overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Total Cost",
                value=f"${cost_data['total_cost']:.4f}"
            )
            
        with col2:
            st.metric(
                label="Total Tokens",
                value=f"{cost_data['total_tokens']:,}"
            )
            
        with col3:
            avg_cost_per_query = cost_data['total_cost'] / max(cost_data['total_queries'], 1)
            st.metric(
                label="Avg Cost per Query",
                value=f"${avg_cost_per_query:.6f}"
            )
            
        # Cost breakdown by model
        if cost_data.get('cost_by_model'):
            model_cost_df = pd.DataFrame(
                cost_data['cost_by_model'],
                columns=['model', 'total_cost', 'total_tokens', 'query_count']
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_cost_by_model = px.pie(
                    model_cost_df,
                    values='total_cost',
                    names='model',
                    title='Cost Distribution by Model'
                )
                st.plotly_chart(fig_cost_by_model, use_container_width=True)
                
            with col2:
                fig_tokens_by_model = px.bar(
                    model_cost_df,
                    x='model',
                    y='total_tokens',
                    title='Token Usage by Model',
                    labels={'total_tokens': 'Total Tokens'}
                )
                st.plotly_chart(fig_tokens_by_model, use_container_width=True)
                
        # Hourly cost trends
        if cost_data.get('hourly_costs'):
            hourly_cost_df = pd.DataFrame(
                cost_data['hourly_costs'],
                columns=['hour', 'total_cost', 'total_tokens']
            )
            
            st.subheader("📈 Cost Trends")
            
            fig_hourly_cost = px.line(
                hourly_cost_df,
                x='hour',
                y='total_cost',
                title='Hourly Cost Trends',
                labels={'hour': 'Hour of Day', 'total_cost': 'Cost ($)'}
            )
            st.plotly_chart(fig_hourly_cost, use_container_width=True)
            
        # Detailed cost table
        st.subheader("📋 Detailed Cost Breakdown")
        st.dataframe(model_cost_df, use_container_width=True)
        
    def _render_performance_alerts(self):
        """Render performance alerts and system health"""
        st.header("🚨 Performance Alerts")
        
        time_range = st.session_state.get('time_range_hours', 24)
        alerts = self.system_monitor.get_performance_alerts(hours=time_range)
        
        if not alerts:
            st.success("✅ No performance alerts in the selected time range. System is running normally.")
            return
            
        # Alert summary
        critical_alerts = [a for a in alerts if a['severity'] == 'critical']
        warning_alerts = [a for a in alerts if a['severity'] == 'warning']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="🔴 Critical Alerts",
                value=len(critical_alerts)
            )
            
        with col2:  
            st.metric(
                label="🟡 Warning Alerts",
                value=len(warning_alerts)
            )
            
        with col3:
            acknowledged_alerts = [a for a in alerts if a['acknowledged']]
            st.metric(
                label="✅ Acknowledged",
                value=len(acknowledged_alerts)
            )
            
        # Alert details
        st.subheader("Alert Details")
        
        for alert in alerts[:10]:  # Show latest 10 alerts
            severity_color = {
                'critical': 'alert-critical',
                'warning': 'alert-warning'
            }.get(alert['severity'], 'alert-warning')
            
            st.markdown(f"""
            <div class="{severity_color}">
                <strong>{alert['alert_type'].upper()}</strong> - {alert['timestamp']}<br>
                {alert['metric_name']}: {alert['current_value']:.2f} (threshold: {alert['threshold_value']:.2f})
            </div>
            """, unsafe_allow_html=True)

# Global dashboard instance
analytics_dashboard = AnalyticsDashboard()