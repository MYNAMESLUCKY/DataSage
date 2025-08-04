"""
Enterprise System Monitoring
Real-time system metrics collection and monitoring
"""

import psutil
import time
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import threading
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    active_connections: int
    gpu_usage: float = 0.0
    temperature: float = 0.0

@dataclass
class QueryMetrics:
    query_id: str
    timestamp: datetime
    query_text: str
    processing_time: float
    model_used: str
    success: bool
    token_count: int
    cost_estimate: float
    user_id: str
    response_quality_score: float

class SystemMonitor:
    """Real-time system monitoring and metrics collection"""
    
    def __init__(self, db_path: str = "analytics.db"):
        self.db_path = db_path
        self.monitoring = False
        self.monitor_thread = None
        self._init_database()
        
    def _init_database(self):
        """Initialize analytics database"""
        with sqlite3.connect(self.db_path) as conn:
            # System metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    cpu_percent REAL NOT NULL,
                    memory_percent REAL NOT NULL,
                    memory_used_mb REAL NOT NULL,
                    memory_available_mb REAL NOT NULL,
                    disk_usage_percent REAL NOT NULL,
                    disk_free_gb REAL NOT NULL,
                    active_connections INTEGER DEFAULT 0,
                    gpu_usage REAL DEFAULT 0.0,
                    temperature REAL DEFAULT 0.0
                )
            """)
            
            # Query metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    query_text TEXT NOT NULL,
                    processing_time REAL NOT NULL,
                    model_used TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    token_count INTEGER DEFAULT 0,
                    cost_estimate REAL DEFAULT 0.0,
                    user_id TEXT DEFAULT 'anonymous',
                    response_quality_score REAL DEFAULT 0.0
                )
            """)
            
            # Error tracking table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS error_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    error_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    component TEXT NOT NULL,
                    severity TEXT DEFAULT 'error',
                    resolved BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Performance alerts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    alert_type TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    threshold_value REAL NOT NULL,
                    severity TEXT DEFAULT 'warning',
                    acknowledged BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_system_timestamp ON system_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query_timestamp ON query_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query_user ON query_metrics(user_id)")
            
    def start_monitoring(self, interval: int = 30):
        """Start real-time system monitoring"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info(f"System monitoring started with {interval}s interval")
        
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("System monitoring stopped")
        
    def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                metrics = self._collect_system_metrics()
                self._store_system_metrics(metrics)
                self._check_performance_alerts(metrics)
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(interval)
                
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Network connections (approximate active connections)
        try:
            connections = len(psutil.net_connections())
        except:
            connections = 0
            
        # GPU usage (if available)
        gpu_usage = 0.0
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = gpus[0].load * 100
        except:
            pass
            
        # Temperature (if available)
        temperature = 0.0
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Get CPU temperature if available
                for name, entries in temps.items():
                    if entries:
                        temperature = entries[0].current
                        break
        except:
            pass
            
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            disk_usage_percent=disk.percent,
            disk_free_gb=disk.free / (1024 * 1024 * 1024),
            active_connections=connections,
            gpu_usage=gpu_usage,
            temperature=temperature
        )
        
    def _store_system_metrics(self, metrics: SystemMetrics):
        """Store system metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO system_metrics 
                    (timestamp, cpu_percent, memory_percent, memory_used_mb, 
                     memory_available_mb, disk_usage_percent, disk_free_gb, 
                     active_connections, gpu_usage, temperature)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.timestamp, metrics.cpu_percent, metrics.memory_percent,
                    metrics.memory_used_mb, metrics.memory_available_mb,
                    metrics.disk_usage_percent, metrics.disk_free_gb,
                    metrics.active_connections, metrics.gpu_usage, metrics.temperature
                ))
        except Exception as e:
            logger.error(f"Failed to store system metrics: {e}")
            
    def _check_performance_alerts(self, metrics: SystemMetrics):
        """Check for performance alerts and store them"""
        alerts = []
        
        # CPU alert
        if metrics.cpu_percent > 80:
            alerts.append(("cpu_high", "cpu_percent", metrics.cpu_percent, 80, "warning"))
        elif metrics.cpu_percent > 95:
            alerts.append(("cpu_critical", "cpu_percent", metrics.cpu_percent, 95, "critical"))
            
        # Memory alert
        if metrics.memory_percent > 85:
            alerts.append(("memory_high", "memory_percent", metrics.memory_percent, 85, "warning"))
        elif metrics.memory_percent > 95:
            alerts.append(("memory_critical", "memory_percent", metrics.memory_percent, 95, "critical"))
            
        # Disk alert
        if metrics.disk_usage_percent > 90:
            alerts.append(("disk_high", "disk_usage_percent", metrics.disk_usage_percent, 90, "warning"))
            
        # Store alerts
        if alerts:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    for alert_type, metric_name, current_value, threshold, severity in alerts:
                        conn.execute("""
                            INSERT INTO performance_alerts 
                            (timestamp, alert_type, metric_name, current_value, threshold_value, severity)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (datetime.now(), alert_type, metric_name, current_value, threshold, severity))
            except Exception as e:
                logger.error(f"Failed to store performance alerts: {e}")
                
    def record_query_metrics(self, query_metrics: QueryMetrics):
        """Record query performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO query_metrics 
                    (query_id, timestamp, query_text, processing_time, model_used, 
                     success, token_count, cost_estimate, user_id, response_quality_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    query_metrics.query_id, query_metrics.timestamp, query_metrics.query_text,
                    query_metrics.processing_time, query_metrics.model_used, query_metrics.success,
                    query_metrics.token_count, query_metrics.cost_estimate, query_metrics.user_id,
                    query_metrics.response_quality_score
                ))
        except Exception as e:
            logger.error(f"Failed to record query metrics: {e}")
            
    def record_error(self, error_type: str, error_message: str, component: str, severity: str = "error"):
        """Record system errors"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO error_log (timestamp, error_type, error_message, component, severity)
                    VALUES (?, ?, ?, ?, ?)
                """, (datetime.now(), error_type, error_message, component, severity))
        except Exception as e:
            logger.error(f"Failed to record error: {e}")
            
    def get_system_metrics(self, hours: int = 24) -> List[Dict]:
        """Get system metrics for the last N hours"""
        start_time = datetime.now() - timedelta(hours=hours)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT timestamp, cpu_percent, memory_percent, memory_used_mb,
                           disk_usage_percent, active_connections, gpu_usage, temperature
                    FROM system_metrics 
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                """, (start_time,))
                
                metrics = []
                for row in cursor.fetchall():
                    metrics.append({
                        'timestamp': row[0],
                        'cpu_percent': row[1],
                        'memory_percent': row[2],
                        'memory_used_mb': row[3],
                        'disk_usage_percent': row[4],
                        'active_connections': row[5],
                        'gpu_usage': row[6],
                        'temperature': row[7]
                    })
                return metrics
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return []
            
    def get_query_analytics(self, hours: int = 24) -> Dict:
        """Get query analytics for the last N hours"""
        start_time = datetime.now() - timedelta(hours=hours)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total queries
                cursor = conn.execute("""
                    SELECT COUNT(*), AVG(processing_time), AVG(response_quality_score),
                           SUM(CASE WHEN success THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
                    FROM query_metrics WHERE timestamp > ?
                """, (start_time,))
                
                total_stats = cursor.fetchone()
                
                # Query patterns by hour
                cursor = conn.execute("""
                    SELECT strftime('%H', timestamp) as hour, COUNT(*), AVG(processing_time)
                    FROM query_metrics 
                    WHERE timestamp > ?
                    GROUP BY strftime('%H', timestamp)
                    ORDER BY hour
                """, (start_time,))
                
                hourly_patterns = cursor.fetchall()
                
                # Model usage
                cursor = conn.execute("""
                    SELECT model_used, COUNT(*), AVG(processing_time), AVG(cost_estimate)
                    FROM query_metrics 
                    WHERE timestamp > ?
                    GROUP BY model_used
                    ORDER BY COUNT(*) DESC
                """, (start_time,))
                
                model_usage = cursor.fetchall()
                
                # Top users
                cursor = conn.execute("""
                    SELECT user_id, COUNT(*), AVG(processing_time)
                    FROM query_metrics 
                    WHERE timestamp > ? AND user_id != 'anonymous'
                    GROUP BY user_id
                    ORDER BY COUNT(*) DESC
                    LIMIT 10
                """, (start_time,))
                
                top_users = cursor.fetchall()
                
                return {
                    'total_queries': total_stats[0] or 0,
                    'avg_processing_time': total_stats[1] or 0,
                    'avg_quality_score': total_stats[2] or 0,
                    'success_rate': total_stats[3] or 0,
                    'hourly_patterns': hourly_patterns,
                    'model_usage': model_usage,
                    'top_users': top_users
                }
        except Exception as e:
            logger.error(f"Failed to get query analytics: {e}")
            return {}
            
    def get_cost_analytics(self, hours: int = 24) -> Dict:
        """Get cost analytics for the last N hours"""
        start_time = datetime.now() - timedelta(hours=hours)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT SUM(cost_estimate), SUM(token_count), model_used,
                           COUNT(*) as query_count
                    FROM query_metrics 
                    WHERE timestamp > ?
                    GROUP BY model_used
                    ORDER BY SUM(cost_estimate) DESC
                """, (start_time,))
                
                cost_by_model = cursor.fetchall()
                
                # Hourly cost trends
                cursor = conn.execute("""
                    SELECT strftime('%H', timestamp) as hour, 
                           SUM(cost_estimate), SUM(token_count)
                    FROM query_metrics 
                    WHERE timestamp > ?
                    GROUP BY strftime('%H', timestamp)
                    ORDER BY hour
                """, (start_time,))
                
                hourly_costs = cursor.fetchall()
                
                # Total costs
                cursor = conn.execute("""
                    SELECT SUM(cost_estimate), SUM(token_count), COUNT(*)
                    FROM query_metrics 
                    WHERE timestamp > ?
                """, (start_time,))
                
                total_stats = cursor.fetchone()
                
                return {
                    'total_cost': total_stats[0] or 0,
                    'total_tokens': total_stats[1] or 0,
                    'total_queries': total_stats[2] or 0,
                    'cost_by_model': cost_by_model,
                    'hourly_costs': hourly_costs
                }
        except Exception as e:
            logger.error(f"Failed to get cost analytics: {e}")
            return {}
            
    def get_performance_alerts(self, hours: int = 24) -> List[Dict]:
        """Get performance alerts for the last N hours"""
        start_time = datetime.now() - timedelta(hours=hours)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT timestamp, alert_type, metric_name, current_value, 
                           threshold_value, severity, acknowledged
                    FROM performance_alerts 
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                """, (start_time,))
                
                alerts = []
                for row in cursor.fetchall():
                    alerts.append({
                        'timestamp': row[0],
                        'alert_type': row[1],
                        'metric_name': row[2],
                        'current_value': row[3],
                        'threshold_value': row[4],
                        'severity': row[5],
                        'acknowledged': row[6]
                    })
                return alerts
        except Exception as e:
            logger.error(f"Failed to get performance alerts: {e}")
            return []
            
    def cleanup_old_data(self, days: int = 30):
        """Clean up old analytics data"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Keep only recent data
                conn.execute("DELETE FROM system_metrics WHERE timestamp < ?", (cutoff_date,))
                conn.execute("DELETE FROM query_metrics WHERE timestamp < ?", (cutoff_date,))
                conn.execute("DELETE FROM error_log WHERE timestamp < ?", (cutoff_date,))
                conn.execute("DELETE FROM performance_alerts WHERE timestamp < ?", (cutoff_date,))
                
                # Vacuum database to reclaim space
                conn.execute("VACUUM")
                
            logger.info(f"Cleaned up analytics data older than {days} days")
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")

# Global system monitor instance
system_monitor = SystemMonitor()