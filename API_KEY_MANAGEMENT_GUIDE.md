# 🔑 API Key Management System - Complete Guide

## ✅ System Overview

The One-Click API Key Generation and Management system provides enterprise-grade security and convenience for managing programmatic access to your RAG system. This feature enables users to create, manage, monitor, and control API keys directly from both the web interface and REST API.

## 🚀 Key Features

### 🔐 Secure Key Generation
- **One-click creation** with customizable settings
- **Cryptographically secure** keys using `secrets.token_urlsafe(32)`
- **SHA-256 hashing** for secure storage
- **Unique prefixes** (`rag_`) for easy identification

### 🎛️ Comprehensive Management
- **Multiple access scopes** (read-only, query-only, ingest-only, full-access, admin)
- **Custom expiration** settings (1-365 days or never)
- **Rate limiting** configuration (1-10,000 requests/hour)
- **Key revocation** and regeneration
- **Metadata management** (name, description)

### 📊 Usage Analytics
- **Real-time usage tracking** with detailed statistics
- **Success rate monitoring** and error tracking
- **Endpoint usage analysis** showing most-used features
- **Daily usage charts** for trend analysis
- **Historical data** retention for up to 365 days

### 🛡️ Enterprise Security
- **SQLite database** for secure key storage
- **JWT authentication** for API access
- **IP-based rate limiting** to prevent abuse
- **Audit logging** for all key operations
- **Secure key display** (only shown once)

## 📡 API Endpoints

All API key management endpoints are available at `/api-keys/*`:

### Authentication
```bash
# Get access token
curl -X POST "http://localhost:8000/auth/token?user_id=YOUR_USER_ID&role=user"
```

### Key Management
```bash
# List all keys
GET /api-keys/list

# Generate new key  
POST /api-keys/generate
{
  "name": "My API Key",
  "description": "For production app",
  "scope": "query_only",
  "expires_in_days": 90,
  "rate_limit": 1000
}

# Get key details
GET /api-keys/{key_id}

# Update key metadata
PUT /api-keys/{key_id}
{
  "name": "Updated name",
  "description": "Updated description"
}

# Revoke key
DELETE /api-keys/{key_id}

# Regenerate key
POST /api-keys/{key_id}/regenerate

# Get usage statistics
GET /api-keys/{key_id}/usage?days=30

# Get available scopes
GET /api-keys/scopes/available
```

## 🖥️ Web Interface

### Streamlit Integration
The API Key Management system is fully integrated into the main Streamlit application with a dedicated "🔑 API Keys" tab that provides:

1. **My API Keys Tab**
   - Visual cards showing all user keys
   - Status indicators (🟢 active, 🔴 expired, ⚫ revoked)
   - Usage metrics and last-used timestamps
   - One-click revoke and regenerate buttons

2. **Generate New Key Tab**
   - User-friendly form with validation
   - Scope selection with descriptions
   - Advanced settings (expiration, rate limits)
   - Instant key generation with secure display
   - Copy-paste ready code examples

3. **Usage Analytics Tab**
   - Key selection dropdown
   - Time period slider (7-90 days)
   - Usage metrics dashboard
   - Endpoint popularity charts
   - Success rate monitoring

## 🔧 Technical Implementation

### Database Schema
```sql
-- API Keys table
CREATE TABLE api_keys (
    key_id TEXT PRIMARY KEY,
    key_prefix TEXT NOT NULL,
    key_hash TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    description TEXT,
    user_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP,
    last_used TIMESTAMP,
    usage_count INTEGER DEFAULT 0,
    rate_limit INTEGER DEFAULT 100,
    rate_window INTEGER DEFAULT 3600,
    metadata TEXT
);

-- Usage logs table
CREATE TABLE key_usage_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key_id TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    ip_address TEXT,
    user_agent TEXT,
    success BOOLEAN NOT NULL,
    error_message TEXT
);
```

### Access Scopes
- **`read_only`**: System information and health endpoints only
- **`query_only`**: Knowledge base queries and search operations
- **`ingest_only`**: Document upload and data ingestion only
- **`full_access`**: Complete query and ingest capabilities
- **`admin`**: All system operations (requires admin role)

### Security Features
- **Hash-based storage**: Keys are never stored in plain text
- **Rate limiting**: Per-key limits prevent API abuse
- **Automatic expiration**: Keys can be set to expire automatically
- **Audit trails**: All operations are logged for security
- **User isolation**: Users can only manage their own keys

## 📝 Usage Examples

### Python Client Integration
```python
import requests

class RAGAPIClient:
    def __init__(self, api_key, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def query(self, question):
        response = requests.post(
            f"{self.base_url}/query",
            headers=self.headers,
            json={"query": question}
        )
        return response.json()
    
    def ingest_text(self, text, metadata=None):
        response = requests.post(
            f"{self.base_url}/ingest",
            headers=self.headers,
            json={
                "data_type": "text",
                "content": text,
                "metadata": metadata or {}
            }
        )
        return response.json()

# Usage
client = RAGAPIClient("rag_your_api_key_here")
result = client.query("What is machine learning?")
print(result['answer'])
```

### Enterprise Integration
```python
# For enterprise applications
class EnterpriseRAGClient:
    def __init__(self, api_key, base_url, retry_attempts=3):
        self.client = RAGAPIClient(api_key, base_url)
        self.retry_attempts = retry_attempts
    
    def robust_query(self, question):
        for attempt in range(self.retry_attempts):
            try:
                return self.client.query(question)
            except requests.exceptions.RateLimitError:
                if attempt < self.retry_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
```

### Webhook Integration
```python
# Send query results to Slack
def send_to_slack(query_result, webhook_url):
    payload = {
        "text": f"RAG Query Result",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Query:* {query_result['query']}\n*Answer:* {query_result['answer'][:200]}..."
                }
            }
        ]
    }
    requests.post(webhook_url, json=payload)
```

## 🎯 Best Practices

### Key Management
1. **Use descriptive names** for easy identification
2. **Set appropriate scopes** based on application needs
3. **Configure rate limits** to match usage patterns
4. **Set expiration dates** for temporary access
5. **Regularly review** and revoke unused keys
6. **Monitor usage patterns** for security anomalies

### Security Recommendations
1. **Store keys securely** in environment variables or secret managers
2. **Never commit keys** to version control
3. **Use HTTPS** in production environments
4. **Implement retry logic** with exponential backoff
5. **Monitor rate limits** to avoid 429 errors
6. **Rotate keys regularly** for enhanced security

### Integration Guidelines
1. **Handle errors gracefully** with proper exception handling
2. **Implement logging** for debugging and monitoring
3. **Use connection pooling** for high-volume applications
4. **Cache responses** when appropriate to reduce API calls
5. **Validate API responses** before processing data

## 📊 Monitoring & Analytics

### Key Metrics to Track
- **Usage frequency**: Requests per day/hour
- **Success rates**: Error vs. successful requests
- **Response times**: API performance monitoring
- **Endpoint popularity**: Most-used features
- **User patterns**: Access times and frequency

### Alerts & Notifications
- **High error rates**: Above 5% failure rate
- **Rate limit hits**: Approaching or exceeding limits
- **Unusual usage patterns**: Sudden spikes or suspicious activity
- **Key expiration**: Warnings before keys expire
- **Security events**: Multiple failed authentications

## 🚀 Future Enhancements

### Planned Features
1. **API Key Templates**: Pre-configured settings for common use cases
2. **Team Management**: Shared keys for organizations
3. **Advanced Analytics**: Machine learning insights for usage patterns
4. **Integration Marketplace**: Pre-built connectors for popular services
5. **Automated Rotation**: Scheduled key regeneration
6. **Compliance Reporting**: SOC2 and GDPR compliance dashboards

### Integration Roadmap
1. **Slack Bot**: Direct Slack commands for key management
2. **CLI Tool**: Command-line interface for developers
3. **Dashboard Widgets**: Embeddable usage charts
4. **Mobile App**: iOS/Android key management
5. **SSO Integration**: Enterprise identity providers

## ✅ System Status

### ✅ Completed Features
- ✅ Secure key generation and storage
- ✅ Complete REST API endpoints
- ✅ Streamlit web interface integration
- ✅ Usage tracking and analytics
- ✅ Multiple access scopes
- ✅ Rate limiting and security
- ✅ Key lifecycle management
- ✅ Comprehensive documentation

### 🎯 Ready for Production
The API Key Management system is **production-ready** with:
- Enterprise-grade security
- Comprehensive monitoring
- User-friendly interface
- Complete API documentation
- Robust error handling
- Scalable architecture

The system successfully enables one-click API key generation and management, providing enterprise users with secure, convenient access to programmatic RAG system capabilities.