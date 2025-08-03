# Enterprise RAG System - User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Account Registration](#account-registration)
3. [System Features](#system-features)
4. [Query System](#query-system)
5. [File Processing](#file-processing)
6. [Analytics Dashboard](#analytics-dashboard)
7. [Security Features](#security-features)
8. [Troubleshooting](#troubleshooting)

## Getting Started

### Accessing the System
1. Navigate to your deployed Enterprise RAG System URL
2. You'll be presented with a secure login interface
3. Register a new account or login with existing credentials

### First Time Setup
1. **Register Account**: Click "Register" to create your account
   - Choose a username (minimum 3 characters)
   - Provide a valid email address
   - Create a secure password (minimum 8 characters)
   - Select appropriate user role (User is default)

2. **Login**: After registration, login with your credentials
   - The system has rate limiting: 5 login attempts per 5 minutes
   - Failed attempts are tracked for security

## Account Registration

### Registration Process
1. Click the **"Register"** button on the login page
2. Fill in the registration form:
   - **Username**: Unique identifier (3+ characters)
   - **Email**: Valid email address for notifications
   - **Password**: Secure password (8+ characters)
   - **Confirm Password**: Must match the password
3. Click **"Create Account"**
4. Upon success, you'll be redirected to login

### User Roles
- **Admin**: Full system access, user management, system configuration
- **User**: Standard access to all RAG features, file uploads, queries
- **Viewer**: Read-only access to system information and analytics

## System Features

### Main Interface
The system provides a professional tabbed interface:

1. **Query System**: Ask questions and get intelligent answers
2. **File Processing**: Upload and process documents
3. **Analytics**: View system performance and usage statistics
4. **System Info**: Technical details and configuration

### Navigation
- **Top Navigation**: Tabs for different system sections
- **Sidebar**: User information, logout, and quick actions
- **Status Indicators**: Real-time system health and security status

## Query System

### How to Ask Questions
1. Navigate to the **"Query System"** tab
2. Enter your question in the text area
3. Configure query settings (optional):
   - **AI Model**: Choose from available models (SARVAM-M recommended)
   - **Max Results**: Number of sources to consider (default: 20)
   - **Web Search**: Enable real-time web search integration
4. Click **"Process Query"**

### Query Features
- **Intelligent Hybrid RAG**: Combines knowledge base with real-time web search
- **Auto Knowledge Updates**: System automatically adds new information from web
- **Multiple AI Providers**: Automatic fallback between SARVAM, DeepSeek, and OpenAI
- **Source Attribution**: All answers include source references
- **Copy Functionality**: Easy copying of answers with dedicated text area

### Query Types
- **Factual Questions**: "What are the benefits of renewable energy?"
- **Comparative Analysis**: "Compare solar vs wind energy efficiency"
- **Research Queries**: "Latest developments in AI technology 2024"
- **Complex Analysis**: Multi-part questions with detailed requirements

### Best Practices
- Be specific in your questions for better results
- Use clear, well-formed questions
- Take advantage of web search for current information
- Review source citations for credibility

## File Processing

### Supported File Types
- **Text Files**: .txt, .md
- **Documents**: .pdf, .docx
- **Spreadsheets**: .xlsx, .csv
- **Web Content**: URLs for automatic scraping

### Upload Process
1. Go to **"File Processing"** tab
2. Choose upload method:
   - **File Upload**: Drag and drop or browse files
   - **URL Input**: Enter web URLs for scraping
   - **Text Input**: Direct text entry
3. Click **"Process Files"**
4. Monitor processing status in real-time

### Processing Features
- **Intelligent Chunking**: Documents split optimally for AI processing
- **Metadata Extraction**: Automatic source attribution and categorization
- **Vector Storage**: Documents converted to searchable embeddings
- **Batch Processing**: Multiple files processed simultaneously
- **Progress Tracking**: Real-time processing status updates

### Rate Limits
- **File Uploads**: 10 files per hour per user
- **Processing**: Automatic queue management
- **Storage**: Unlimited document storage in vector database

## Analytics Dashboard

### Performance Metrics
- **Response Times**: Average query processing speeds
- **Knowledge Base Growth**: Document count over time
- **User Activity**: Query patterns and usage statistics
- **System Health**: Real-time operational status

### Available Charts
- **Query Performance**: Response time trends
- **Usage Statistics**: Most popular features
- **Knowledge Base Analytics**: Document types and sources
- **Security Metrics**: Login attempts and rate limiting stats

### Exporting Data
- **CSV Export**: Download analytics data
- **Report Generation**: Automated system reports
- **Custom Timeframes**: Filter data by date ranges

## Security Features

### Authentication
- **JWT Tokens**: Secure, time-limited session tokens
- **Password Security**: PBKDF2 hashing with individual salts
- **Session Management**: Automatic token expiration and cleanup
- **Brute Force Protection**: Account lockout after failed attempts

### Rate Limiting
- **Query Limits**: 50 queries per hour per user
- **Login Limits**: 5 attempts per 5 minutes
- **Upload Limits**: 10 files per hour
- **API Limits**: 100 calls per hour for integrations

### Data Protection
- **Input Validation**: All user inputs validated and sanitized
- **SQL Injection Prevention**: Parameterized queries only
- **XSS Protection**: HTML content properly escaped
- **Secure Storage**: Encrypted database connections

### Browser Security
- **Developer Tools Disabled**: F12, inspect element blocked
- **Right-Click Disabled**: Context menu access prevented
- **Console Protection**: Regular console clearing and warnings
- **Text Selection Limited**: Prevents easy content copying

### Monitoring
- **Activity Logging**: All user actions logged
- **Security Events**: Failed logins and suspicious activity tracked
- **Rate Limit Tracking**: Automatic blocking of excessive requests
- **System Health**: Continuous monitoring of all components

## Troubleshooting

### Common Issues

#### Login Problems
- **Forgot Password**: Contact administrator for password reset
- **Account Locked**: Wait 5 minutes after failed login attempts
- **Rate Limited**: Reduce login frequency, wait for limit reset

#### Query Issues
- **No Results**: Try rephrasing question or enabling web search
- **Slow Responses**: Check system load in analytics dashboard
- **Rate Limit Reached**: Wait for hourly limit reset (50 queries/hour)

#### File Upload Problems
- **File Too Large**: Check file size limits with administrator
- **Unsupported Format**: Use supported file types (.txt, .pdf, .docx, .xlsx, .csv)
- **Upload Failed**: Check internet connection and retry

#### Performance Issues
- **Slow Loading**: Check network connection and system status
- **Timeout Errors**: Large queries may take longer, be patient
- **Browser Compatibility**: Use modern browsers (Chrome, Firefox, Safari, Edge)

### Getting Help
- **System Status**: Check analytics dashboard for operational status
- **Contact Administrator**: For account issues or technical problems
- **Rate Limit Info**: Current limits displayed in security section
- **Documentation**: Refer to this guide for detailed instructions

### Security Best Practices
- **Strong Passwords**: Use complex passwords with mixed characters
- **Regular Logout**: Always logout when finished
- **Secure Network**: Use secure internet connections
- **Keep Credentials Private**: Never share login information
- **Report Issues**: Immediately report suspicious activity

### System Requirements
- **Browser**: Modern web browser with JavaScript enabled
- **Internet**: Stable internet connection for real-time features
- **Screen Resolution**: Minimum 1024x768 for optimal experience
- **JavaScript**: Must be enabled for full functionality

### Contact Information
- **Technical Support**: Contact your system administrator
- **Security Issues**: Report immediately to security team
- **Feature Requests**: Submit through appropriate channels
- **Training**: Additional training materials available on request

---

## Quick Reference

### Keyboard Shortcuts
- **Ctrl+Enter**: Submit query (in query text area)
- **Tab**: Navigate between form fields
- **Esc**: Close modal dialogs

### User Limits
- **Queries**: 50 per hour
- **Logins**: 5 attempts per 5 minutes  
- **Uploads**: 10 files per hour
- **Session**: 24 hours maximum

### File Size Limits
- **Individual Files**: Check with administrator
- **Total Storage**: Unlimited vector database storage
- **Batch Uploads**: 10 files maximum per batch

### Response Times
- **Simple Queries**: 1-3 seconds
- **Complex Queries**: 3-10 seconds
- **File Processing**: Varies by file size
- **Web Search**: 2-5 additional seconds

This guide covers all essential features and operations of the Enterprise RAG System. For additional support or advanced features, contact your system administrator.