# Firebase Google Authentication Setup Guide

This guide explains how to configure Google Firebase authentication for the Enterprise RAG System.

## Overview

The system now supports dual authentication methods:
- **Google Firebase Authentication**: Sign in with Google accounts
- **Standard Authentication**: Username/password registration and login

When Firebase is configured, users will see both options in a tabbed interface. Without Firebase configuration, only standard authentication is available.

## Required Firebase Configuration

### 1. Firebase Project Setup

1. **Create Firebase Project**
   - Go to [Firebase Console](https://console.firebase.google.com/)
   - Click "Create a project" or select existing project
   - Enable Google Analytics (optional)
   - Complete project creation

2. **Enable Authentication**
   - In Firebase Console, go to "Authentication"
   - Click "Get started"
   - Go to "Sign-in method" tab
   - Enable "Google" provider
   - Set up OAuth consent screen if prompted

### 2. Service Account Credentials

1. **Generate Service Account**
   - Go to Project Settings (gear icon)
   - Click "Service accounts" tab
   - Click "Generate new private key"
   - Download JSON file containing credentials

2. **Extract Required Values from JSON:**
   ```json
   {
     "project_id": "your-project-id",
     "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
     "client_email": "firebase-adminsdk-xxx@your-project.iam.gserviceaccount.com"
   }
   ```

### 3. Google OAuth Configuration

1. **Google Cloud Console Setup**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Select your Firebase project
   - Navigate to "APIs & Services" > "Credentials"

2. **Create OAuth 2.0 Client**
   - Click "Create Credentials" > "OAuth 2.0 Client ID"
   - Choose "Web application"
   - Set name (e.g., "Enterprise RAG System")
   - Add authorized redirect URIs:
     - `http://localhost:5000/auth/callback` (development)
     - `https://your-replit-domain.replit.app/auth/callback` (production)

3. **Download OAuth Credentials**
   - Download the client configuration JSON
   - Extract `client_id` and `client_secret`

## Environment Variables Configuration

Add these secrets to your Replit environment or `.env` file:

```bash
# Firebase Service Account
FIREBASE_PROJECT_ID=your-firebase-project-id
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_HERE\n-----END PRIVATE KEY-----\n"
FIREBASE_CLIENT_EMAIL=firebase-adminsdk-xxx@your-project.iam.gserviceaccount.com

# Google OAuth
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-client-secret
```

**Important Notes:**
- Keep the `\n` characters in `FIREBASE_PRIVATE_KEY` - they are required
- Enclose the private key in quotes if it contains special characters
- All values are case-sensitive

## System Integration

### Authentication Flow

1. **User Access**: User visits the login page
2. **Option Selection**: User chooses between Google Login or Standard Login tabs
3. **Google Authentication**:
   - User clicks "Sign in with Google"
   - Redirected to Google OAuth consent screen
   - After consent, redirected back with authorization code
   - System exchanges code for access token
   - User information retrieved from Google
   - Local user record created/updated
   - JWT token generated for session

### User Role Assignment

- **First User**: Automatically assigned `admin` role
- **Existing Users**: Retain their existing role from local database
- **New Users**: Assigned `user` role by default
- **Role Changes**: Admins can modify roles through admin panel

### Local Database Synchronization

Firebase users are automatically synchronized with the local authentication database:
- Email used as username
- Special password hash: `firebase:{firebase_uid}`
- Special salt: `firebase_auth`
- Role preserved from previous sessions
- Standard JWT tokens issued for consistency

## Features

### Dual Authentication Support
- Seamless switching between Google and standard login
- Consistent JWT token system for both methods
- Unified user management in admin panel
- Same rate limiting and security features

### Security Benefits
- OAuth 2.0 standard compliance
- Google's enterprise-grade authentication
- Automatic email verification through Google
- No password storage for Google users
- Same security protections as standard auth

### User Experience
- Single sign-on with Google accounts
- Profile pictures from Google accounts
- Automatic role assignment
- Clean tabbed interface
- Fallback to standard auth if Google unavailable

## Testing Authentication

### Without Firebase Configuration
- Only "Standard Login" tab visible
- Users must register manually
- Username/password authentication only

### With Firebase Configuration
- Both "Google Login" and "Standard Login" tabs visible
- Users can choose authentication method
- Google users automatically registered
- Admin panel shows authentication provider

### Verification Commands
```bash
# Test Firebase availability
python -c "
import os
firebase_vars = ['FIREBASE_PROJECT_ID', 'FIREBASE_PRIVATE_KEY', 'FIREBASE_CLIENT_EMAIL', 'GOOGLE_CLIENT_ID', 'GOOGLE_CLIENT_SECRET']
available = all(os.getenv(var) for var in firebase_vars)
print('Firebase Available:', available)
"

# Test Firebase initialization
python -c "
import sys; sys.path.append('src')
from src.auth.firebase_auth import FirebaseAuthManager
firebase = FirebaseAuthManager()
print('Firebase Ready:', firebase.is_firebase_available())
"
```

## Troubleshooting

### Common Issues

**"Firebase not configured" message**
- Check all 5 environment variables are set
- Verify private key format includes `\n` characters
- Ensure no extra spaces in variable values

**"Google OAuth configuration not available"**
- Verify `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` are set
- Check OAuth client is configured in Google Cloud Console
- Ensure redirect URIs match your deployment URL

**"Failed to initialize Firebase"**
- Verify `FIREBASE_PRIVATE_KEY` format is correct
- Check service account has proper permissions
- Ensure project ID matches Firebase project

**OAuth redirect errors**
- Update authorized redirect URIs in Google Cloud Console
- Include both development and production URLs
- Format: `https://your-domain/auth/callback`

### Debug Steps

1. **Check Environment Variables**
   ```bash
   echo $FIREBASE_PROJECT_ID
   echo $GOOGLE_CLIENT_ID
   # (Don't echo private keys in production)
   ```

2. **Test Firebase Connection**
   - Use the verification commands above
   - Check application logs for Firebase initialization messages

3. **Verify OAuth Configuration**
   - Test Google OAuth URL generation
   - Check redirect URI configuration
   - Verify client ID/secret in Google Cloud Console

## Production Deployment

### Security Considerations
- Use HTTPS in production (automatic with Replit deployment)
- Update redirect URIs to production domains
- Store secrets securely (use Replit Secrets, not code)
- Monitor authentication logs for suspicious activity

### Performance Notes
- Firebase tokens cached for session duration
- Local database updated only on first Google login
- Standard JWT tokens used for all subsequent requests
- No additional Firebase calls after initial authentication

## Support

For Firebase-specific issues, refer to:
- [Firebase Authentication Documentation](https://firebase.google.com/docs/auth)
- [Google OAuth 2.0 Documentation](https://developers.google.com/identity/protocols/oauth2)
- [Google Cloud Console](https://console.cloud.google.com/)

For system integration issues, check:
- Application logs for specific error messages
- Admin panel for user management and authentication statistics
- Local authentication database for user synchronization status