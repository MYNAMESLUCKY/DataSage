#!/usr/bin/env python3
"""
Enterprise integrations for RAG system
Supports CRM, ERP, and business application connections
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import aiohttp
import base64
import os

from ..utils.utils import setup_logging

logger = setup_logging(__name__)

class SalesforceIntegration:
    """Salesforce CRM integration"""
    
    def __init__(self):
        self.instance_url = os.getenv('SALESFORCE_INSTANCE_URL')
        self.client_id = os.getenv('SALESFORCE_CLIENT_ID')
        self.client_secret = os.getenv('SALESFORCE_CLIENT_SECRET')
        self.username = os.getenv('SALESFORCE_USERNAME')
        self.password = os.getenv('SALESFORCE_PASSWORD')
        self.security_token = os.getenv('SALESFORCE_SECURITY_TOKEN')
        self.access_token = None
        self.token_expires_at = None
    
    async def authenticate(self) -> bool:
        """Authenticate with Salesforce"""
        try:
            auth_url = f"{self.instance_url}/services/oauth2/token"
            
            data = {
                'grant_type': 'password',
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'username': self.username,
                'password': f"{self.password}{self.security_token}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(auth_url, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.access_token = result['access_token']
                        self.instance_url = result['instance_url']
                        # Token expires in ~2 hours
                        self.token_expires_at = datetime.utcnow() + timedelta(hours=1, minutes=50)
                        logger.info("Salesforce authentication successful")
                        return True
                    else:
                        logger.error(f"Salesforce auth failed: {await response.text()}")
                        return False
                        
        except Exception as e:
            logger.error(f"Salesforce authentication error: {e}")
            return False
    
    async def _ensure_authenticated(self):
        """Ensure valid authentication token"""
        if not self.access_token or (self.token_expires_at and datetime.utcnow() >= self.token_expires_at):
            await self.authenticate()
    
    async def search_knowledge_articles(self, query: str) -> List[Dict[str, Any]]:
        """Search Salesforce knowledge articles"""
        await self._ensure_authenticated()
        
        try:
            search_url = f"{self.instance_url}/services/data/v58.0/search/"
            
            # SOSL query for knowledge articles
            sosl_query = f"FIND {{'{query}'}} IN ALL FIELDS RETURNING KnowledgeArticleVersion(Id, Title, Summary, ArticleBody WHERE PublishStatus='Online')"
            
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            
            params = {'q': sosl_query}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=headers, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        articles = []
                        
                        for search_record in result.get('searchRecords', []):
                            articles.append({
                                'id': search_record['Id'],
                                'title': search_record.get('Title', ''),
                                'summary': search_record.get('Summary', ''),
                                'body': search_record.get('ArticleBody', ''),
                                'source': 'Salesforce Knowledge'
                            })
                        
                        return articles
                    else:
                        logger.error(f"Salesforce search failed: {await response.text()}")
                        return []
                        
        except Exception as e:
            logger.error(f"Salesforce knowledge search error: {e}")
            return []
    
    async def create_case_from_query(self, query: str, user_email: str) -> Optional[str]:
        """Create a support case from a complex query"""
        await self._ensure_authenticated()
        
        try:
            case_url = f"{self.instance_url}/services/data/v58.0/sobjects/Case/"
            
            case_data = {
                'Subject': f"RAG System Query: {query[:100]}...",
                'Description': f"Complex query from RAG system:\n\nQuery: {query}\nRequested by: {user_email}\nTimestamp: {datetime.utcnow().isoformat()}",
                'Origin': 'RAG System',
                'Priority': 'Medium',
                'Status': 'New'
            }
            
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(case_url, headers=headers, json=case_data) as response:
                    if response.status == 201:
                        result = await response.json()
                        case_id = result['id']
                        logger.info(f"Created Salesforce case: {case_id}")
                        return case_id
                    else:
                        logger.error(f"Case creation failed: {await response.text()}")
                        return None
                        
        except Exception as e:
            logger.error(f"Salesforce case creation error: {e}")
            return None

class Office365Integration:
    """Microsoft 365 integration"""
    
    def __init__(self):
        self.tenant_id = os.getenv('M365_TENANT_ID')
        self.client_id = os.getenv('M365_CLIENT_ID')
        self.client_secret = os.getenv('M365_CLIENT_SECRET')
        self.access_token = None
        self.token_expires_at = None
    
    async def authenticate(self) -> bool:
        """Authenticate with Microsoft Graph API"""
        try:
            auth_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
            
            data = {
                'grant_type': 'client_credentials',
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'scope': 'https://graph.microsoft.com/.default'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(auth_url, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.access_token = result['access_token']
                        expires_in = result.get('expires_in', 3600)
                        self.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in - 300)
                        logger.info("Microsoft 365 authentication successful")
                        return True
                    else:
                        logger.error(f"M365 auth failed: {await response.text()}")
                        return False
                        
        except Exception as e:
            logger.error(f"M365 authentication error: {e}")
            return False
    
    async def _ensure_authenticated(self):
        """Ensure valid authentication token"""
        if not self.access_token or (self.token_expires_at and datetime.utcnow() >= self.token_expires_at):
            await self.authenticate()
    
    async def search_sharepoint_documents(self, query: str, site_id: str = None) -> List[Dict[str, Any]]:
        """Search SharePoint documents"""
        await self._ensure_authenticated()
        
        try:
            if site_id:
                search_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/search(q='{query}')"
            else:
                search_url = f"https://graph.microsoft.com/v1.0/me/drive/search(q='{query}')"
            
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        documents = []
                        
                        for item in result.get('value', []):
                            documents.append({
                                'id': item['id'],
                                'name': item['name'],
                                'url': item['webUrl'],
                                'last_modified': item.get('lastModifiedDateTime', ''),
                                'size': item.get('size', 0),
                                'source': 'SharePoint'
                            })
                        
                        return documents
                    else:
                        logger.error(f"SharePoint search failed: {await response.text()}")
                        return []
                        
        except Exception as e:
            logger.error(f"SharePoint search error: {e}")
            return []
    
    async def send_teams_message(self, channel_id: str, message: str) -> bool:
        """Send message to Teams channel"""
        await self._ensure_authenticated()
        
        try:
            message_url = f"https://graph.microsoft.com/v1.0/teams/{channel_id}/channels/general/messages"
            
            message_data = {
                "body": {
                    "content": message,
                    "contentType": "text"
                }
            }
            
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(message_url, headers=headers, json=message_data) as response:
                    if response.status == 201:
                        logger.info("Teams message sent successfully")
                        return True
                    else:
                        logger.error(f"Teams message failed: {await response.text()}")
                        return False
                        
        except Exception as e:
            logger.error(f"Teams message error: {e}")
            return False

class GoogleWorkspaceIntegration:
    """Google Workspace integration"""
    
    def __init__(self):
        self.service_account_file = os.getenv('GOOGLE_SERVICE_ACCOUNT_FILE')
        self.domain = os.getenv('GOOGLE_WORKSPACE_DOMAIN')
    
    async def search_drive_documents(self, query: str) -> List[Dict[str, Any]]:
        """Search Google Drive documents"""
        try:
            # This would use Google Drive API
            # Implementation depends on google-api-python-client
            logger.info(f"Searching Google Drive for: {query}")
            
            # Placeholder implementation
            return [{
                'id': 'placeholder',
                'name': f'Document matching: {query}',
                'url': 'https://drive.google.com/...',
                'source': 'Google Drive'
            }]
            
        except Exception as e:
            logger.error(f"Google Drive search error: {e}")
            return []
    
    async def search_gmail_threads(self, query: str, user_email: str) -> List[Dict[str, Any]]:
        """Search Gmail threads for relevant information"""
        try:
            # This would use Gmail API
            logger.info(f"Searching Gmail for: {query}")
            
            # Placeholder implementation
            return [{
                'id': 'placeholder',
                'subject': f'Email thread about: {query}',
                'snippet': 'Email content snippet...',
                'source': 'Gmail'
            }]
            
        except Exception as e:
            logger.error(f"Gmail search error: {e}")
            return []

class ZendeskIntegration:
    """Zendesk support integration"""
    
    def __init__(self):
        self.subdomain = os.getenv('ZENDESK_SUBDOMAIN')
        self.email = os.getenv('ZENDESK_EMAIL')
        self.api_token = os.getenv('ZENDESK_API_TOKEN')
        self.base_url = f"https://{self.subdomain}.zendesk.com/api/v2"
    
    async def search_articles(self, query: str) -> List[Dict[str, Any]]:
        """Search Zendesk knowledge base articles"""
        try:
            search_url = f"{self.base_url}/help_center/articles/search.json"
            
            auth_string = base64.b64encode(f"{self.email}/token:{self.api_token}".encode()).decode()
            headers = {
                'Authorization': f'Basic {auth_string}',
                'Content-Type': 'application/json'
            }
            
            params = {'query': query}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=headers, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        articles = []
                        
                        for article in result.get('results', []):
                            articles.append({
                                'id': article['id'],
                                'title': article['title'],
                                'body': article.get('body', ''),
                                'url': article['html_url'],
                                'source': 'Zendesk'
                            })
                        
                        return articles
                    else:
                        logger.error(f"Zendesk search failed: {await response.text()}")
                        return []
                        
        except Exception as e:
            logger.error(f"Zendesk search error: {e}")
            return []
    
    async def create_ticket(self, subject: str, description: str, requester_email: str) -> Optional[str]:
        """Create a support ticket from complex query"""
        try:
            ticket_url = f"{self.base_url}/tickets.json"
            
            ticket_data = {
                "ticket": {
                    "subject": subject,
                    "comment": {
                        "body": description
                    },
                    "requester": {
                        "email": requester_email
                    },
                    "priority": "normal"
                }
            }
            
            auth_string = base64.b64encode(f"{self.email}/token:{self.api_token}".encode()).decode()
            headers = {
                'Authorization': f'Basic {auth_string}',
                'Content-Type': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(ticket_url, headers=headers, json=ticket_data) as response:
                    if response.status == 201:
                        result = await response.json()
                        ticket_id = result['ticket']['id']
                        logger.info(f"Created Zendesk ticket: {ticket_id}")
                        return str(ticket_id)
                    else:
                        logger.error(f"Zendesk ticket creation failed: {await response.text()}")
                        return None
                        
        except Exception as e:
            logger.error(f"Zendesk ticket creation error: {e}")
            return None

class IntegrationManager:
    """Manages all enterprise integrations"""
    
    def __init__(self):
        self.salesforce = SalesforceIntegration()
        self.office365 = Office365Integration()
        self.google_workspace = GoogleWorkspaceIntegration()
        self.zendesk = ZendeskIntegration()
    
    async def search_all_systems(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """Search across all integrated systems"""
        results = {}
        
        # Run searches in parallel
        tasks = []
        
        if self.salesforce.instance_url:
            tasks.append(('salesforce', self.salesforce.search_knowledge_articles(query)))
        
        if self.office365.tenant_id:
            tasks.append(('sharepoint', self.office365.search_sharepoint_documents(query)))
        
        if self.google_workspace.service_account_file:
            tasks.append(('google_drive', self.google_workspace.search_drive_documents(query)))
        
        if self.zendesk.subdomain:
            tasks.append(('zendesk', self.zendesk.search_articles(query)))
        
        # Execute all searches concurrently
        if tasks:
            search_results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
            
            for i, (system_name, _) in enumerate(tasks):
                result = search_results[i]
                if isinstance(result, Exception):
                    logger.error(f"Search failed for {system_name}: {result}")
                    results[system_name] = []
                else:
                    results[system_name] = result
        
        return results
    
    async def escalate_complex_query(self, query: str, user_email: str) -> Dict[str, Optional[str]]:
        """Escalate complex queries to support systems"""
        escalation_results = {}
        
        # Create tickets/cases in parallel
        tasks = []
        
        if self.salesforce.instance_url:
            tasks.append(('salesforce', self.salesforce.create_case_from_query(query, user_email)))
        
        if self.zendesk.subdomain:
            tasks.append(('zendesk', self.zendesk.create_ticket(
                f"Complex RAG Query: {query[:50]}...",
                f"Complex query from RAG system:\n\nQuery: {query}\nUser: {user_email}",
                user_email
            )))
        
        if tasks:
            results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
            
            for i, (system_name, _) in enumerate(tasks):
                result = results[i]
                if isinstance(result, Exception):
                    logger.error(f"Escalation failed for {system_name}: {result}")
                    escalation_results[system_name] = None
                else:
                    escalation_results[system_name] = result
        
        return escalation_results

# Global integration manager
integration_manager = IntegrationManager()