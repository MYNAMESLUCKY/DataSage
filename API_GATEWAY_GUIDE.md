# 🚀 Enterprise RAG API Gateway Guide

The Enterprise RAG API Gateway provides REST API access to your intelligent knowledge retrieval system. It enables programmatic access, enterprise integrations, and scalable query processing.

## 🔗 API Endpoints

### Base URL
- **Local Development**: `http://localhost:8000`
- **Production**: `https://your-domain.replit.app`

### Documentation
- **Interactive API Docs**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`

## 🔐 Authentication

### Get Access Token
```bash
curl -X POST "http://localhost:8000/auth/token?user_id=your_user_id&role=user" \
  -H "accept: application/json"
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer"
}
```

### Using the Token
Include the token in the Authorization header for all API calls:
```bash
-H "Authorization: Bearer your_access_token_here"
```

## 📊 Core API Endpoints

### 1. Health Check
Check system status and component health.

```bash
curl -X GET "http://localhost:8000/health" \
  -H "accept: application/json"
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-03T21:30:00.000Z",
  "version": "1.0.0",
  "components": {
    "api_gateway": "healthy",
    "rag_engine": "healthy",
    "hybrid_processor": "healthy",
    "database": "healthy"
  },
  "rag_system_status": "healthy"
}
```

### 2. Query Knowledge Base
Ask questions and get intelligent responses from the RAG system.

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is quantum computing?",
    "max_results": 10,
    "use_web_search": true,
    "llm_model": "sarvam-m"
  }'
```

**Request Parameters:**
- `query` (required): Your question (1-1000 characters)
- `max_results` (optional): Maximum sources to retrieve (1-20, default: 10)
- `similarity_threshold` (optional): Minimum similarity score (0.0-1.0, default: 0.1)
- `use_web_search` (optional): Enable real-time web search (default: true)
- `llm_model` (optional): AI model to use (default: "sarvam-m")

**Response:**
```json
{
  "answer": "Quantum computing is a revolutionary computing paradigm...",
  "confidence": 0.92,
  "sources": [
    "Document about quantum mechanics",
    "Research paper on quantum algorithms"
  ],
  "web_sources": [
    {
      "title": "Quantum Computing Explained",
      "url": "https://example.com/quantum",
      "snippet": "Recent advances in quantum..."
    }
  ],
  "processing_time": 2.45,
  "model_used": "sarvam-m",
  "api_provider": "SARVAM",
  "status": "success",
  "query_id": "q_1754257890_1234"
}
```

### 3. Data Ingestion
Add new content to the knowledge base.

**Ingest URL:**
```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "data_type": "url",
    "content": "https://example.com/article",
    "metadata": {
      "source": "company_blog",
      "category": "technical"
    }
  }'
```

**Ingest Text:**
```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "data_type": "text",
    "content": "This is important company information...",
    "metadata": {
      "department": "HR",
      "document_type": "policy"
    }
  }'
```

**Response:**
```json
{
  "message": "Text ingestion started",
  "status": "processing"
}
```

### 4. System Statistics
Get system metrics and usage statistics.

```bash
curl -X GET "http://localhost:8000/stats" \
  -H "Authorization: Bearer your_token"
```

**Response:**
```json
{
  "total_documents": 3577,
  "query_count_24h": 245,
  "avg_response_time": 2.3,
  "cache_hit_rate": 0.78,
  "system_uptime": "2d 14h 30m",
  "rag_system_status": true
}
```

### 5. Available Models
List supported AI models.

```bash
curl -X GET "http://localhost:8000/models" \
  -H "Authorization: Bearer your_token"
```

**Response:**
```json
{
  "models": [
    {
      "id": "sarvam-m",
      "name": "SARVAM Model",
      "provider": "SARVAM",
      "status": "active",
      "capabilities": ["text_generation", "reasoning"]
    }
  ]
}
```

## 🐍 Python Client Example

```python
import requests
import json

class RAGAPIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.token = None
        self.session = requests.Session()
    
    def authenticate(self, user_id: str, role: str = "user"):
        """Get authentication token"""
        response = self.session.post(
            f"{self.base_url}/auth/token",
            params={"user_id": user_id, "role": role}
        )
        response.raise_for_status()
        
        token_data = response.json()
        self.token = token_data['access_token']
        self.session.headers.update({
            'Authorization': f'Bearer {self.token}'
        })
        return self.token
    
    def query(self, question: str, **kwargs):
        """Query the knowledge base"""
        data = {"query": question, **kwargs}
        response = self.session.post(
            f"{self.base_url}/query", 
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def ingest_text(self, text: str, metadata: dict = None):
        """Add text to knowledge base"""
        data = {
            "data_type": "text",
            "content": text,
            "metadata": metadata or {}
        }
        response = self.session.post(
            f"{self.base_url}/ingest",
            json=data
        )
        response.raise_for_status()
        return response.json()

# Usage example
client = RAGAPIClient()
client.authenticate("demo_user")

# Ask a question
result = client.query(
    "What is machine learning?",
    max_results=5,
    use_web_search=True
)
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.1%}")
```

## 🔌 Enterprise Integrations

### Slack Integration
Use webhooks and slash commands to integrate with Slack:

```bash
# Configure webhook in environment
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"

# Send query result to Slack channel
curl -X POST "$SLACK_WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "RAG Query Result",
    "blocks": [
      {
        "type": "section",
        "text": {
          "type": "mrkdwn",
          "text": "*Question:* What is AI?\n*Answer:* Artificial Intelligence is..."
        }
      }
    ]
  }'
```

### Microsoft Teams Integration
```bash
# Configure Teams webhook
export TEAMS_WEBHOOK_URL="https://your-org.webhook.office.com/..."

# Send adaptive card to Teams
curl -X POST "$TEAMS_WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "@type": "MessageCard",
    "@context": "http://schema.org/extensions",
    "themeColor": "0078D4",
    "summary": "RAG Query Result",
    "sections": [{
      "activityTitle": "Knowledge Base Query",
      "text": "Answer: Artificial Intelligence is..."
    }]
  }'
```

## 🛡️ Security Features

### Rate Limiting
- **Default Limit**: 100 requests per hour per IP
- **Response**: 429 Too Many Requests when exceeded
- **Headers**: Rate limit info in response headers

### JWT Authentication
- **Algorithm**: HS256
- **Expiry**: 24 hours (configurable)
- **Claims**: user ID, role, expiration

### CORS Configuration
- **Development**: Allow all origins
- **Production**: Configure specific domains

## 🔧 Configuration

### Environment Variables
```bash
# Security
export JWT_SECRET="your-secure-secret-key"

# API Configuration
export API_HOST="0.0.0.0"
export API_PORT="8000"
export API_RELOAD="false"

# Integrations
export SLACK_WEBHOOK_URL="https://hooks.slack.com/..."
export TEAMS_WEBHOOK_URL="https://your-org.webhook.office.com/..."

# RAG System
export SARVAM_API="your_sarvam_api_key"
export TAVILY_API_KEY="your_tavily_api_key"
export DATABASE_URL="your_postgresql_url"
```

## 📈 Monitoring & Analytics

### Query Analytics
All queries are logged with:
- Query ID and timestamp
- User ID and processing time
- Model used and confidence score
- Success/failure status

### Health Monitoring
Regular health checks provide:
- Component status (healthy/unhealthy)
- System uptime and version
- RAG system initialization status

### Performance Metrics
- Average response time
- Query volume (24h)
- Cache hit rate
- Document count

## 🚀 Deployment

### Development
```bash
# Start API Gateway
python api_gateway_simple.py

# Access documentation
open http://localhost:8000/docs
```

### Production
```bash
# Using uvicorn directly
uvicorn api_gateway_simple:app --host 0.0.0.0 --port 8000 --workers 4

# Using gunicorn
gunicorn api_gateway_simple:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "api_gateway_simple:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🐛 Troubleshooting

### Common Issues

**1. Authentication Errors**
```bash
# Check token validity
curl -X GET "http://localhost:8000/health" \
  -H "Authorization: Bearer your_token"
```

**2. Rate Limit Exceeded**
```json
{
  "error": "Rate limit exceeded. Try again later.",
  "status_code": 429
}
```
**Solution**: Wait or implement exponential backoff

**3. RAG System Unavailable**
```json
{
  "answer": "RAG system is currently initializing...",
  "status": "fallback"
}
```
**Solution**: Check RAG system logs and configuration

### Debug Mode
Enable detailed logging:
```bash
export LOG_LEVEL="DEBUG"
python api_gateway_simple.py
```

## 📝 API Response Codes

- **200**: Success
- **201**: Resource created
- **400**: Bad request (invalid parameters)
- **401**: Unauthorized (invalid/missing token)
- **403**: Forbidden (insufficient permissions)
- **429**: Rate limit exceeded
- **500**: Internal server error
- **503**: Service unavailable (RAG system down)

## 🔗 Next Steps

1. **Set up webhooks** for Slack/Teams integration
2. **Configure monitoring** with Prometheus/Grafana
3. **Implement caching** with Redis for better performance
4. **Add API versioning** for backward compatibility
5. **Set up load balancing** for high availability

The API Gateway is now ready for enterprise use with comprehensive REST endpoints, authentication, and integration capabilities!