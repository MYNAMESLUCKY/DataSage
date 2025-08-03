#!/usr/bin/env python3
"""
API Key Management UI Components for Streamlit
Provides user interface for managing API keys within the main application
"""

import streamlit as st
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class APIKeyUI:
    """UI components for API key management"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url.rstrip('/')
    
    def render_api_key_management(self, user_token: str):
        """Render the main API key management interface"""
        
        st.markdown("## 🔑 API Key Management")
        st.markdown("Generate and manage API keys for programmatic access to your RAG system.")
        
        # Tabs for different sections
        tab1, tab2, tab3 = st.tabs(["📋 My API Keys", "➕ Generate New Key", "📊 Usage Analytics"])
        
        with tab1:
            self._render_key_list(user_token)
        
        with tab2:
            self._render_key_generation(user_token)
        
        with tab3:
            self._render_usage_analytics(user_token)
    
    def _render_key_list(self, user_token: str):
        """Render the list of existing API keys"""
        st.markdown("### Your API Keys")
        
        try:
            # Fetch user's API keys
            response = requests.get(
                f"{self.api_base_url}/api-keys/list",
                headers={"Authorization": f"Bearer {user_token}"}
            )
            
            if response.status_code == 200:
                keys = response.json()
                
                if not keys:
                    st.info("You haven't created any API keys yet. Use the 'Generate New Key' tab to create one.")
                    return
                
                for key in keys:
                    self._render_key_card(key, user_token)
                    
            else:
                st.error(f"Failed to fetch API keys: {response.text}")
                
        except Exception as e:
            st.error(f"Error fetching API keys: {str(e)}")
    
    def _render_key_card(self, key: Dict[str, Any], user_token: str):
        """Render a single API key card"""
        
        # Status indicator
        status_colors = {
            "active": "🟢",
            "suspended": "🟡",
            "expired": "🔴",
            "revoked": "⚫"
        }
        
        status_icon = status_colors.get(key['status'], "❓")
        
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{status_icon} {key['name']}**")
                st.caption(f"ID: {key['key_id'][:12]}... | Scope: {key['scope']}")
                if key['description']:
                    st.caption(f"Description: {key['description']}")
            
            with col2:
                st.metric("Usage", key['usage_count'])
                if key['last_used']:
                    last_used = datetime.fromisoformat(key['last_used'].replace('Z', '+00:00'))
                    st.caption(f"Last used: {last_used.strftime('%Y-%m-%d')}")
                else:
                    st.caption("Never used")
            
            with col3:
                if key['status'] == 'active':
                    if st.button("Revoke", key=f"revoke_{key['key_id']}", type="secondary"):
                        self._revoke_key(key['key_id'], user_token)
                        st.rerun()
                    
                    if st.button("Regenerate", key=f"regen_{key['key_id']}", type="primary"):
                        self._regenerate_key(key['key_id'], user_token)
                        st.rerun()
                else:
                    st.caption(f"Status: {key['status']}")
            
            # Expandable details
            with st.expander("Details"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.text(f"Created: {key['created_at'][:10]}")
                    st.text(f"Rate limit: {key['rate_limit']}/hour")
                    
                with col2:
                    if key['expires_at']:
                        st.text(f"Expires: {key['expires_at'][:10]}")
                    else:
                        st.text("Expires: Never")
                    st.text(f"Status: {key['status']}")
                
                # Usage button
                if st.button("View Usage Stats", key=f"stats_{key['key_id']}"):
                    self._show_key_stats(key['key_id'], user_token)
            
            st.divider()
    
    def _render_key_generation(self, user_token: str):
        """Render the API key generation form"""
        st.markdown("### Generate New API Key")
        
        with st.form("generate_key_form"):
            # Basic information
            name = st.text_input(
                "Key Name *",
                placeholder="e.g., 'Production App API Key'",
                help="A descriptive name to identify this key"
            )
            
            description = st.text_area(
                "Description",
                placeholder="Optional description of what this key will be used for",
                height=100
            )
            
            # Scope selection
            scope_options = {
                "read_only": "Read Only - System information and health endpoints",
                "query_only": "Query Only - Search and retrieve information",
                "ingest_only": "Ingest Only - Add documents to knowledge base",
                "full_access": "Full Access - Query and ingest operations",
                "admin": "Admin - All system operations (requires admin role)"
            }
            
            scope = st.selectbox(
                "Access Scope *",
                options=list(scope_options.keys()),
                format_func=lambda x: scope_options[x],
                index=1  # Default to query_only
            )
            
            # Advanced settings
            with st.expander("Advanced Settings"):
                col1, col2 = st.columns(2)
                
                with col1:
                    expires_in_days = st.number_input(
                        "Expires in days",
                        min_value=0,
                        max_value=365,
                        value=0,
                        help="0 = Never expires"
                    )
                
                with col2:
                    rate_limit = st.number_input(
                        "Rate limit (requests/hour)",
                        min_value=1,
                        max_value=10000,
                        value=100
                    )
            
            submitted = st.form_submit_button("🔑 Generate API Key", type="primary")
            
            if submitted:
                if not name.strip():
                    st.error("Key name is required")
                else:
                    self._generate_new_key(
                        name=name.strip(),
                        description=description.strip(),
                        scope=scope,
                        expires_in_days=expires_in_days if expires_in_days > 0 else None,
                        rate_limit=rate_limit,
                        user_token=user_token
                    )
    
    def _render_usage_analytics(self, user_token: str):
        """Render usage analytics for all user's keys"""
        st.markdown("### Usage Analytics")
        
        try:
            # Get user's keys
            response = requests.get(
                f"{self.api_base_url}/api-keys/list",
                headers={"Authorization": f"Bearer {user_token}"}
            )
            
            if response.status_code == 200:
                keys = response.json()
                active_keys = [k for k in keys if k['status'] == 'active']
                
                if not active_keys:
                    st.info("No active API keys to show analytics for.")
                    return
                
                # Key selector
                selected_key = st.selectbox(
                    "Select API Key",
                    options=active_keys,
                    format_func=lambda k: f"{k['name']} ({k['usage_count']} uses)"
                )
                
                if selected_key:
                    self._show_key_stats(selected_key['key_id'], user_token)
                    
        except Exception as e:
            st.error(f"Error loading analytics: {str(e)}")
    
    def _generate_new_key(self, name: str, description: str, scope: str, 
                         expires_in_days: Optional[int], rate_limit: int, user_token: str):
        """Generate a new API key"""
        try:
            payload = {
                "name": name,
                "description": description,
                "scope": scope,
                "rate_limit": rate_limit
            }
            
            if expires_in_days:
                payload["expires_in_days"] = expires_in_days
            
            response = requests.post(
                f"{self.api_base_url}/api-keys/generate",
                headers={"Authorization": f"Bearer {user_token}"},
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Show success with the actual API key
                st.success("✅ API Key Generated Successfully!")
                
                # Display the key in a copyable format
                st.markdown("### 🔐 Your New API Key")
                st.code(result['api_key'])
                
                st.warning("⚠️ **Important**: This is the only time you'll see this key. Store it securely!")
                
                # Show key details
                with st.expander("Key Details"):
                    key_info = result['key_info']
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.text(f"Name: {key_info['name']}")
                        st.text(f"Scope: {key_info['scope']}")
                        st.text(f"Rate Limit: {key_info['rate_limit']}/hour")
                    
                    with col2:
                        st.text(f"Created: {key_info['created_at'][:16]}")
                        if key_info['expires_at']:
                            st.text(f"Expires: {key_info['expires_at'][:10]}")
                        else:
                            st.text("Expires: Never")
                
                # Usage example
                with st.expander("Usage Example"):
                    st.markdown("### Python Example")
                    st.code(f"""
import requests

# Using your new API key
headers = {{
    "Authorization": "Bearer {result['api_key']}",
    "Content-Type": "application/json"
}}

# Query the knowledge base
response = requests.post(
    "{self.api_base_url}/query",
    headers=headers,
    json={{"query": "What is artificial intelligence?"}}
)

result = response.json()
print(result['answer'])
                    """, language="python")
                
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                st.error(f"Failed to generate API key: {error_detail}")
                
        except Exception as e:
            st.error(f"Error generating API key: {str(e)}")
    
    def _revoke_key(self, key_id: str, user_token: str):
        """Revoke an API key"""
        try:
            response = requests.delete(
                f"{self.api_base_url}/api-keys/{key_id}",
                headers={"Authorization": f"Bearer {user_token}"}
            )
            
            if response.status_code == 200:
                st.success("API key revoked successfully")
            else:
                st.error(f"Failed to revoke key: {response.text}")
                
        except Exception as e:
            st.error(f"Error revoking key: {str(e)}")
    
    def _regenerate_key(self, key_id: str, user_token: str):
        """Regenerate an API key"""
        try:
            response = requests.post(
                f"{self.api_base_url}/api-keys/{key_id}/regenerate",
                headers={"Authorization": f"Bearer {user_token}"}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                st.success("✅ API Key Regenerated Successfully!")
                st.markdown("### 🔐 Your New API Key")
                st.code(result['api_key'])
                st.warning("⚠️ The old key has been revoked. Update your applications with this new key.")
                
            else:
                st.error(f"Failed to regenerate key: {response.text}")
                
        except Exception as e:
            st.error(f"Error regenerating key: {str(e)}")
    
    def _show_key_stats(self, key_id: str, user_token: str):
        """Show usage statistics for a key"""
        try:
            days = st.select_slider(
                "Time period",
                options=[7, 14, 30, 60, 90],
                value=30,
                format_func=lambda x: f"Last {x} days"
            )
            
            response = requests.get(
                f"{self.api_base_url}/api-keys/{key_id}/usage",
                headers={"Authorization": f"Bearer {user_token}"},
                params={"days": days}
            )
            
            if response.status_code == 200:
                stats = response.json()
                
                # Overview metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Requests", stats['total_requests'])
                
                with col2:
                    st.metric("Success Rate", f"{stats['success_rate']:.1f}%")
                
                with col3:
                    st.metric("Successful Requests", stats['successful_requests'])
                
                # Top endpoints
                if stats['top_endpoints']:
                    st.markdown("#### Most Used Endpoints")
                    for endpoint in stats['top_endpoints']:
                        st.text(f"{endpoint['endpoint']}: {endpoint['count']} requests")
                
                # Daily usage chart (if available)
                if stats['daily_usage']:
                    st.markdown("#### Daily Usage")
                    # Simple text display (could be enhanced with actual charts)
                    for day in stats['daily_usage'][:7]:  # Show last 7 days
                        st.text(f"{day['date']}: {day['count']} requests")
                
            else:
                st.error("Failed to load usage statistics")
                
        except Exception as e:
            st.error(f"Error loading statistics: {str(e)}")