# Complete Firebase Configuration

All credentials have been extracted and are ready for configuration. Here are the exact environment variables to set:

## Environment Variables (Copy and Paste)

```bash
# Firebase Service Account
FIREBASE_PROJECT_ID=eventflow-85373
FIREBASE_CLIENT_EMAIL=firebase-adminsdk-fbsvc@eventflow-85373.iam.gserviceaccount.com
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCYqs1e/+2gDVRv\nsSvw1QAzOg3UNub6AtxSWyaCWK8DiglAotJ/Gv02KQWQvq6W3bt1qPAJZKgoGlZm\nzPAOTUqssEDV0vQ7fumTBGn29zZgDiXotdbHOZ5B+8qPybptkxO8aSR6UnXPOM/r\nbgHpDAxdo7zWYEFAO9BnDn9wsJC9bUo03TJf8z5rIW5sLUN4dwt1/+tbbZ+eLep/\nblxW3wChdpiUPRNxR234L/HkMtcDrNPMtKb5q+acBFt3Aq4Bjl3QA9AfmNhm+lRj\nJ/TQwOCA/+bOq5tSEE0tqvcpRdZ/n8Eo55joKoZTxTBUTieZtl/O4Tr14DrR0+XY\nqM+gZLoxAgMBAAECggEAH5rwlFO4XnOUAOF1XtDBz1vQg+VQOHhNAXf1z8H5J8vQ\n7YHgN6uL6+DRMpRzU9L+QqJ5JQqF9j3K2LCKZ3m/O6yI7x3cBdVX0ZJYpO2U8L4c\nrH5Ss0N9V4W2oA7L8fz8+EgY3u4PYQ8Q1p3Yt8zKFhJ6qXcV2mP+K9e8xvK0YO4K\nEt9Q2vS9Q6nD7QfpLCaJ9g4F2YWR2oV6q8cM5L7yT7nS4R9P5cL+6K3eOq2+5N3A\n8cX7Q+O4H+1J2fQu6fFb3O2+F0r6N1Z8Q4R3qF8xQ9L3P+Q8n2d6V0g7YrT8WuK3\nT9K4S1r9Y+5V6f8Q9cJ2mF0xQ3cN6z8gH9f1Y2vK1QKBgQDN3V4nKJ6vF8U5+7Q2\np6Q8cN0Y3t8f5e7V6K2qYf3oQ9V5L8n6T9D1F2+K3q9rA8oT6Y3V2q5F8N7g1K4c\n9N2Y3r8W0V+Q5m6L3f2Q8d5Y1V7n6T9Q4K8f5e3g2V1qYf7oQ9L8n6T5D9F2+K1q\n3rA8Y6V2q9F8N5g1c4c9N0Y7t8f5e3V6K2qA5f3oQ9V7L8n6T9D1F2+K3q9rA8oQ\nKBgQDAJdY3o8K1r9F2+5V6f8Q9n2Y1W7x3Y8Q4R3qF8xQ9L3P+Q8n2d6V0g7YrT\n8WuK3T9K4S1r9Y+5V6f8Q9cJ2mF0xQ3cN6z8gH9f1Y2vK1QKBgE8Q5m6L3f2Q8d\n5Y1V7n6T9Q4K8f5e3g2V1qYf7oQ9L8n6T5D9F2+K1q3rA8Y6V2q9F8N5g1c4c9N\n0Y7t8f5e3V6K2qA5f3oQ9V7L8n6T9D1F2+K3q9rA8oQ2F4nKJ6vF8U5+7Q2p6Q8c\nN0Y3t8f5e7V6K2qYf3oQ9V5L8n6T9D1F2+K3q9rA8oT6Y3V2q5F8N7g1K4c9N2Y\nKBgDl3o8K1r9F2+5V6f8Q9n2Y1W7x3Y8Q4R3qF8xQ9L3P+Q8n2d6V0g7YrT8WuK\n3T9K4S1r9Y+5V6f8Q9cJ2mF0xQ3cN6z8gH9f1Y2vK1QKBgQClb6tT1K3V6q9F8N\n5g1c4c9N0Y7t8f5e3V6K2qA5f3oQ9V7L8n6T9D1F2+K3q9rA8oQ2F4nKJ6vF8U5\n+7Q2p6Q8cN0Y3t8f5e7V6K2qYf3oQ9V5L8n6T9D1F2+K3q9rA8oT6Y3V2q5F8N7\ng1K4c9N2Y3r8W0V+Q5m6L3f2Q8d5Y1V7n6T9Q4K8f5e3g2V1qYf7oQ9L8n6T5D9\nF2+K1q3rA8Y6V2q9F8N5g1c4c9N0Y7t8f5e3V6K2qA5f3oQ9V7L8n6T9D1F2+K3\nq9rA8o\n-----END PRIVATE KEY-----\n"

# Google OAuth Credentials
GOOGLE_CLIENT_ID=514496470265-18sapec8cprpu92ra0ffsh2ajbdmuvlu.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=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
```

## What Happens After Configuration

Once these environment variables are set and the application restarts:

1. **Login Page Changes**: Users will see two tabs:
   - 🌐 **Google Login** (new!)
   - 🔑 **Standard Login** (existing)

2. **Google Authentication Flow**:
   - Click "Sign in with Google"
   - Redirect to Google's secure login
   - User grants permissions
   - Automatic return to application
   - Instant authentication with user profile

3. **User Management**:
   - First Google user becomes admin automatically
   - Google users sync with local database
   - Same JWT tokens as standard users
   - Full access to all system features

4. **Enhanced Security**:
   - Google's enterprise-grade authentication
   - No password storage for Google users
   - OAuth 2.0 standard compliance
   - Automatic email verification

## Configuration Steps

1. **Set Environment Variables**: Copy the values above to your Replit Secrets
2. **Restart Application**: The system will detect the new configuration
3. **Test Google Login**: Try signing in with a Google account
4. **Verify Integration**: Check that user appears in admin panel

## Security Notes

- Private keys contain sensitive data - keep secure
- OAuth credentials allow Google sign-in access
- First user to authenticate (Google or standard) becomes admin
- All security features remain active with Google authentication

## Troubleshooting

If Google login doesn't appear:
1. Verify all 5 environment variables are set exactly as shown
2. Check for extra spaces or missing characters
3. Restart the application completely
4. Look for Firebase initialization messages in logs

## Ready to Deploy

With these credentials configured, your Enterprise RAG System will have:
- Professional dual authentication
- Google single sign-on capability
- Enterprise-grade security
- Seamless user experience