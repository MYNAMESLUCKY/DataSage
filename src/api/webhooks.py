#!/usr/bin/env python3
"""
Webhook integrations for Enterprise RAG system
Supports Slack, Microsoft Teams, and custom webhook endpoints
"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import aiohttp
import hashlib
import hmac
import os

from ..utils.utils import setup_logging

logger = setup_logging(__name__)

class WebhookManager:
    """Manages webhook integrations for enterprise systems"""
    
    def __init__(self):
        self.slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        self.teams_webhook_url = os.getenv('TEAMS_WEBHOOK_URL')
        self.custom_webhooks = self._load_custom_webhooks()
        
    def _load_custom_webhooks(self) -> Dict[str, str]:
        """Load custom webhook URLs from environment"""
        webhooks = {}
        webhook_config = os.getenv('CUSTOM_WEBHOOKS', '{}')
        try:
            webhooks = json.loads(webhook_config)
        except json.JSONDecodeError:
            logger.warning("Invalid CUSTOM_WEBHOOKS configuration")
        return webhooks
    
    async def send_slack_notification(self, message: str, channel: str = None, user: str = None):
        """Send notification to Slack"""
        if not self.slack_webhook_url:
            logger.warning("Slack webhook URL not configured")
            return False
            
        payload = {
            "text": message,
            "username": "RAG System",
            "icon_emoji": ":robot_face:",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if channel:
            payload["channel"] = channel
        if user:
            payload["user"] = user
            
        return await self._send_webhook(self.slack_webhook_url, payload)
    
    async def send_teams_notification(self, title: str, message: str, color: str = "0078D4"):
        """Send notification to Microsoft Teams"""
        if not self.teams_webhook_url:
            logger.warning("Teams webhook URL not configured")
            return False
            
        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": color,
            "summary": title,
            "sections": [{
                "activityTitle": title,
                "activitySubtitle": "Enterprise RAG System",
                "activityImage": "https://adaptivecards.io/content/bot-framework.png",
                "text": message,
                "markdown": True
            }]
        }
        
        return await self._send_webhook(self.teams_webhook_url, payload)
    
    async def send_custom_webhook(self, webhook_name: str, data: Dict[str, Any]):
        """Send data to custom webhook endpoint"""
        if webhook_name not in self.custom_webhooks:
            logger.warning(f"Custom webhook '{webhook_name}' not configured")
            return False
            
        webhook_url = self.custom_webhooks[webhook_name]
        return await self._send_webhook(webhook_url, data)
    
    async def _send_webhook(self, url: str, payload: Dict[str, Any]) -> bool:
        """Send webhook with error handling and retry logic"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook sent successfully to {url}")
                        return True
                    else:
                        logger.error(f"Webhook failed with status {response.status}: {await response.text()}")
                        return False
                        
        except Exception as e:
            logger.error(f"Webhook sending failed: {e}")
            return False
    
    async def notify_query_completion(self, query: str, user: str, processing_time: float, status: str):
        """Notify about query completion"""
        if status == "success":
            message = f"✅ Query completed for {user}\nQuery: {query[:100]}...\nTime: {processing_time:.2f}s"
            await self.send_slack_notification(message)
        else:
            message = f"❌ Query failed for {user}\nQuery: {query[:100]}...\nStatus: {status}"
            await self.send_teams_notification("Query Failed", message, "FF0000")
    
    async def notify_system_alert(self, alert_type: str, message: str, severity: str = "warning"):
        """Send system alerts"""
        emoji = "🚨" if severity == "critical" else "⚠️" if severity == "warning" else "ℹ️"
        color = "FF0000" if severity == "critical" else "FFA500" if severity == "warning" else "0078D4"
        
        alert_message = f"{emoji} System Alert: {alert_type}\n{message}"
        
        # Send to both Slack and Teams for critical alerts
        if severity == "critical":
            await asyncio.gather(
                self.send_slack_notification(alert_message),
                self.send_teams_notification(f"Critical Alert: {alert_type}", message, color)
            )
        else:
            await self.send_slack_notification(alert_message)

class SlackBot:
    """Slack bot integration for RAG system"""
    
    def __init__(self, rag_api):
        self.rag_api = rag_api
        self.bot_token = os.getenv('SLACK_BOT_TOKEN')
        self.signing_secret = os.getenv('SLACK_SIGNING_SECRET')
    
    def verify_slack_signature(self, timestamp: str, body: str, signature: str) -> bool:
        """Verify Slack request signature"""
        if not self.signing_secret:
            return False
            
        basestring = f"v0:{timestamp}:{body}"
        expected_signature = 'v0=' + hmac.new(
            self.signing_secret.encode(),
            basestring.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(expected_signature, signature)
    
    async def handle_slash_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Slack slash commands"""
        command = command_data.get('command', '')
        text = command_data.get('text', '')
        user_id = command_data.get('user_id', '')
        
        if command == '/ask':
            return await self._handle_ask_command(text, user_id)
        elif command == '/rag-status':
            return await self._handle_status_command()
        else:
            return {
                "response_type": "ephemeral",
                "text": "Unknown command. Available commands: /ask, /rag-status"
            }
    
    async def _handle_ask_command(self, query: str, user_id: str) -> Dict[str, Any]:
        """Handle /ask command"""
        if not query.strip():
            return {
                "response_type": "ephemeral",
                "text": "Please provide a question. Usage: /ask What is quantum computing?"
            }
        
        try:
            # Process query through RAG system
            result = await self._process_rag_query(query)
            
            return {
                "response_type": "in_channel",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Question:* {query}\n\n*Answer:* {result['answer'][:1000]}..."
                        }
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": f"Confidence: {result['confidence']:.1%} | Sources: {len(result['sources'])} | Time: {result['processing_time']:.2f}s"
                            }
                        ]
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"Slack command processing failed: {e}")
            return {
                "response_type": "ephemeral",
                "text": f"Sorry, I encountered an error processing your question: {str(e)}"
            }
    
    async def _handle_status_command(self) -> Dict[str, Any]:
        """Handle /rag-status command"""
        try:
            # Get system status
            doc_count = self.rag_api.vector_store.get_document_count() if self.rag_api.vector_store else 0
            
            return {
                "response_type": "ephemeral",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*RAG System Status*"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Documents:* {doc_count:,}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Status:* {'🟢 Online' if self.rag_api.rag_engine.is_ready else '🔴 Offline'}"
                            }
                        ]
                    }
                ]
            }
            
        except Exception as e:
            return {
                "response_type": "ephemeral",
                "text": f"Error retrieving status: {str(e)}"
            }
    
    async def _process_rag_query(self, query: str) -> Dict[str, Any]:
        """Process query through RAG system"""
        # This would integrate with your existing RAG processing
        # For now, return a mock response
        return {
            "answer": "This is a placeholder response from the RAG system.",
            "confidence": 0.85,
            "sources": ["doc1", "doc2"],
            "processing_time": 2.5
        }

class TeamsBot:
    """Microsoft Teams bot integration"""
    
    def __init__(self, rag_api):
        self.rag_api = rag_api
        self.app_id = os.getenv('TEAMS_APP_ID')
        self.app_password = os.getenv('TEAMS_APP_PASSWORD')
    
    async def handle_message(self, activity: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Teams message activity"""
        message_text = activity.get('text', '').strip()
        
        if message_text.startswith('/ask'):
            query = message_text[4:].strip()
            return await self._handle_ask_message(query)
        elif message_text.startswith('/status'):
            return await self._handle_status_message()
        else:
            return self._create_help_response()
    
    async def _handle_ask_message(self, query: str) -> Dict[str, Any]:
        """Handle ask message in Teams"""
        if not query:
            return {
                "type": "message",
                "text": "Please provide a question. Usage: /ask What is machine learning?"
            }
        
        try:
            result = await self._process_rag_query(query)
            
            return {
                "type": "message",
                "attachments": [
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": {
                            "type": "AdaptiveCard",
                            "version": "1.2",
                            "body": [
                                {
                                    "type": "TextBlock",
                                    "text": "RAG System Response",
                                    "weight": "Bolder",
                                    "size": "Medium"
                                },
                                {
                                    "type": "TextBlock",
                                    "text": f"**Question:** {query}",
                                    "wrap": True
                                },
                                {
                                    "type": "TextBlock",
                                    "text": f"**Answer:** {result['answer'][:1000]}...",
                                    "wrap": True
                                },
                                {
                                    "type": "FactSet",
                                    "facts": [
                                        {
                                            "title": "Confidence:",
                                            "value": f"{result['confidence']:.1%}"
                                        },
                                        {
                                            "title": "Sources:",
                                            "value": str(len(result['sources']))
                                        },
                                        {
                                            "title": "Processing Time:",
                                            "value": f"{result['processing_time']:.2f}s"
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                ]
            }
            
        except Exception as e:
            return {
                "type": "message",
                "text": f"Error processing question: {str(e)}"
            }
    
    def _create_help_response(self) -> Dict[str, Any]:
        """Create help response for Teams"""
        return {
            "type": "message",
            "text": "Available commands:\n- `/ask [question]` - Ask the RAG system\n- `/status` - Get system status"
        }
    
    async def _process_rag_query(self, query: str) -> Dict[str, Any]:
        """Process query through RAG system"""
        return {
            "answer": "This is a placeholder response from the RAG system.",
            "confidence": 0.85,
            "sources": ["doc1", "doc2"],
            "processing_time": 2.5
        }

# Global webhook manager instance
webhook_manager = WebhookManager()