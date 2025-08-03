# Enterprise RAG System - Administrator Guide

## Overview
This guide provides comprehensive instructions for administrators managing the Enterprise RAG System, including security configuration, user management, and system maintenance.

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Security Configuration](#security-configuration)
3. [User Management](#user-management)
4. [Rate Limiting](#rate-limiting)
5. [Monitoring & Logging](#monitoring--logging)
6. [Database Management](#database-management)
7. [API Configuration](#api-configuration)
8. [Backup & Recovery](#backup--recovery)
9. [Troubleshooting](#troubleshooting)
10. [Security Hardening](#security-hardening)

## System Architecture

### Core Components
- **Frontend**: Streamlit web application with enterprise UI
- **Authentication**: JWT-based with role-based access control
- **Vector Database**: ChromaDB for document storage and similarity search
- **Web Search**: Tavily API integration for real-time information
- **AI Providers**: SARVAM (primary), DeepSeek (fallback), OpenAI (backup)
- **Caching**: PostgreSQL for web search result caching
- **Security**: Multi-layer rate limiting and monitoring

### File Structure
```
enterprise-rag-system/
├── src/
│   ├── auth/                 # Authentication system
│   ├── backend/              # Core RAG functionality
│   ├── components/           # UI components
│   ├── config/               # Configuration management
│   ├── security/             # Rate limiting and security
│   └── utils/                # Utilities and helpers
├── chroma_db/                # Vector database storage
├── enterprise_app.py         # Main application entry
├── USER_GUIDE.md            # User documentation
└── ADMIN_GUIDE.md           # This administrator guide
```

## Security Configuration

### Environment Variables
Ensure these critical environment variables are set:

```bash
# Required API Keys
SARVAM_API=your_sarvam_api_key
DEEPSEEK_API=your_deepseek_api_key
TAVILY_API_KEY=your_tavily_api_key

# Database
DATABASE_URL=your_postgresql_connection_string

# Optional Fallbacks
OPENAI_API_KEY=your_openai_api_key
OPENROUTER_API=your_openrouter_api_key

# Security (Auto-generated if not set)
JWT_SECRET=your_jwt_secret_key
```

### Browser Security Features
The system includes comprehensive browser-based attack prevention:

- **Developer Tools Disabled**: F12, Ctrl+Shift+I, Ctrl+U blocked
- **Right-Click Disabled**: Context menu access prevented
- **Console Protection**: Regular clearing with security warnings
- **Text Selection Limited**: Prevents easy content extraction
- **Streamlit Menu Hidden**: Clean interface without debug options

### Authentication Security
- **Password Hashing**: PBKDF2 with 100,000 iterations + individual salts
- **JWT Tokens**: HS256 algorithm with 24-hour expiration
- **Session Management**: Automatic cleanup of expired sessions
- **Brute Force Protection**: 5 attempts per 5 minutes lockout

## User Management

### Creating Admin Account
Since demo accounts are removed, create the first admin account manually:

```python
# Run this in Python console or script
import sys
sys.path.append('src')
from src.auth.auth_system import AuthenticationSystem, UserRole

auth = AuthenticationSystem()
result = auth.register_user(
    username="admin",
    email="admin@yourcompany.com", 
    password="your_secure_admin_password",
    role=UserRole.ADMIN
)
print(result)
```

### User Roles & Permissions

#### Admin Role
- Full system access and configuration
- User management capabilities
- Rate limit override and management
- System monitoring and analytics
- Security dashboard access

#### User Role
- Standard RAG system access
- Query processing and file uploads
- Personal analytics dashboard
- Limited to standard rate limits

#### Viewer Role
- Read-only access to system information
- View analytics and system status
- Cannot upload files or modify system
- Reduced rate limits

### Managing User Accounts
Access the admin panel after logging in as an admin:

1. Navigate to **Admin Panel** in the sidebar
2. Use **User Management** tab for:
   - View all registered users
   - Modify user roles and permissions
   - Deactivate/activate user accounts
   - Reset rate limits for specific users

## Rate Limiting

### Default Limits
- **Queries**: 50 per hour per user
- **Login Attempts**: 5 per 5 minutes
- **File Uploads**: 10 per hour per user
- **API Calls**: 100 per hour for integrations
- **Registration**: 3 per hour per IP

### Modifying Rate Limits
Edit `src/security/rate_limiter.py`:

```python
self.limits = {
    RateLimitType.QUERY: RateLimit(max_requests=50, window_seconds=3600),
    RateLimitType.LOGIN: RateLimit(max_requests=5, window_seconds=300, block_duration_seconds=900),
    # Modify values as needed
}
```

### Managing Blocked Users
In the admin panel:
1. Go to **Rate Limits** tab
2. View currently blocked identifiers
3. Manually unblock users if needed
4. Monitor rate limit violations

## Monitoring & Logging

### Log Files
Monitor these log areas:
- **Application Logs**: Authentication attempts, query processing
- **Security Logs**: Failed logins, rate limit violations
- **Performance Logs**: Response times, system resource usage
- **Error Logs**: System errors and exceptions

### Key Metrics to Monitor
- **Response Times**: Query processing performance
- **User Activity**: Login patterns and usage statistics
- **Security Events**: Failed authentications and blocked attempts
- **System Resources**: Memory usage, database connections
- **Knowledge Base Growth**: Document count and storage usage

### Setting Up Alerts
Monitor for these critical events:
- Multiple failed login attempts from same IP
- Unusual query patterns or volumes
- System resource exhaustion
- API key errors or rate limits
- Database connection failures

## Database Management

### ChromaDB Vector Store
- **Location**: `chroma_db/` directory
- **Persistence**: Automatic with file-based storage
- **Backup**: Regular backup of entire directory
- **Maintenance**: Automatic cleanup of expired documents

### SQLite Databases
- **Authentication**: `auth.db` - user accounts and sessions
- **Rate Limiting**: `rate_limits.db` - rate limit tracking
- **Backup**: Regular backup of both files
- **Cleanup**: Automatic removal of expired records

### PostgreSQL Cache
- **Purpose**: Web search result caching
- **Configuration**: Via DATABASE_URL environment variable
- **Maintenance**: Automatic cleanup of old cache entries
- **Monitoring**: Track connection pool usage

## API Configuration

### Primary AI Provider (SARVAM)
- **Configuration**: Via SARVAM_API environment variable
- **Model**: sarvam-m for optimal performance
- **Rate Limits**: Handled automatically with fallbacks
- **Monitoring**: Track usage and response times

### Fallback Providers
1. **DeepSeek**: Fast text generation backup
2. **OpenAI**: Final fallback with GPT-4o
3. **OpenRouter**: Alternative routing if configured

### Web Search (Tavily)
- **Configuration**: Via TAVILY_API_KEY
- **Features**: Real-time web search with content cleaning
- **Caching**: Results cached in PostgreSQL
- **Rate Limits**: Managed per API provider limits

## Backup & Recovery

### Critical Data to Backup
1. **Vector Database**: Entire `chroma_db/` directory
2. **User Database**: `auth.db` file
3. **Rate Limit Database**: `rate_limits.db` file
4. **Configuration**: Environment variables and settings
5. **Application Code**: Source code repository

### Backup Schedule
- **Daily**: Database files and vector store
- **Weekly**: Full system backup including code
- **Monthly**: Long-term archive backup
- **Before Updates**: Complete system snapshot

### Recovery Procedures
1. **Database Recovery**: Restore from backup files
2. **Vector Store Recovery**: Replace `chroma_db/` directory
3. **Configuration Recovery**: Restore environment variables
4. **User Session Recovery**: Users will need to re-authenticate

## Troubleshooting

### Common Issues

#### Authentication Problems
- **JWT Secret Missing**: Check JWT_SECRET environment variable
- **Database Locked**: Ensure proper file permissions on auth.db
- **Token Expiration**: Tokens expire after 24 hours (configurable)

#### Performance Issues
- **Slow Queries**: Check vector database size and indexing
- **Memory Usage**: Monitor ChromaDB memory consumption
- **API Timeouts**: Verify AI provider API keys and status

#### Database Issues
- **Connection Errors**: Verify DATABASE_URL for PostgreSQL
- **Disk Space**: Ensure adequate storage for vector database
- **Permissions**: Check file system permissions for SQLite files

### Diagnostic Commands
```bash
# Check system status
python -c "
import sys; sys.path.append('src')
from src.backend.api import RAGSystemAPI
api = RAGSystemAPI()
print(f'Vector store documents: {len(api.vector_store.get_all_documents())}')
print('System operational')
"

# Test authentication
python -c "
import sys; sys.path.append('src')
from src.auth.auth_system import AuthenticationSystem
auth = AuthenticationSystem()
print('Authentication system initialized')
"

# Check rate limiting
python -c "
import sys; sys.path.append('src')
from src.security.rate_limiter import RateLimiter
limiter = RateLimiter()
print('Rate limiting system operational')
"
```

## Security Hardening

### Production Checklist
- [ ] All demo accounts removed
- [ ] Strong JWT_SECRET configured
- [ ] HTTPS enabled (handled by Replit deployment)
- [ ] Rate limiting configured appropriately
- [ ] Database files properly secured
- [ ] API keys stored securely in environment
- [ ] Browser security features enabled
- [ ] Logging and monitoring active
- [ ] Regular backup schedule implemented
- [ ] Error messages don't expose sensitive information

### Network Security
- **HTTPS Only**: All communications encrypted
- **Rate Limiting**: Multiple layers of protection
- **Input Validation**: All user inputs sanitized
- **SQL Injection Prevention**: Parameterized queries only
- **XSS Protection**: HTML content properly escaped

### Access Control
- **Role-Based Permissions**: Strict role enforcement
- **Session Management**: Secure token handling
- **Account Lockout**: Protection against brute force
- **Audit Logging**: All access attempts logged

### Regular Security Tasks
- **Weekly**: Review failed login attempts and security logs
- **Monthly**: Update dependencies and security patches
- **Quarterly**: Full security audit and penetration testing
- **Annually**: Comprehensive security policy review

## Performance Optimization

### Vector Database Optimization
- **Index Maintenance**: Regular cleanup of unused embeddings
- **Memory Management**: Monitor ChromaDB memory usage
- **Query Optimization**: Efficient similarity search parameters
- **Storage Compression**: Optimize vector storage format

### Caching Strategy
- **Web Search Results**: PostgreSQL caching with TTL
- **AI Model Responses**: Response caching for common queries
- **Vector Search**: Embedding cache for frequent searches
- **Session Data**: Efficient session storage and cleanup

### Monitoring Performance
- **Response Times**: Track query processing speeds
- **Resource Usage**: Monitor CPU, memory, and disk usage
- **Database Performance**: Track query execution times
- **API Latency**: Monitor external API response times

## Maintenance Schedule

### Daily Tasks
- Review security logs for unusual activity
- Check system health metrics
- Monitor disk space usage
- Verify backup completion

### Weekly Tasks
- Clean up expired sessions and rate limit records
- Review user activity patterns
- Update system documentation
- Test backup restoration procedures

### Monthly Tasks
- Full system backup and archive
- Security audit and log review
- Performance optimization analysis
- User account cleanup (inactive accounts)

### Quarterly Tasks
- Comprehensive security assessment
- System performance review
- Update dependencies and libraries
- Disaster recovery testing

This guide provides comprehensive coverage of administrative tasks for the Enterprise RAG System. For specific technical issues not covered here, refer to the source code documentation or contact technical support.