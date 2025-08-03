# Quick Firebase Setup for Your Project

Based on your Firebase config, here's what you need to complete the Google authentication setup:

## Your Current Firebase Project Details
- **Project ID**: `eventflow-85373`
- **Auth Domain**: `eventflow-85373.firebaseapp.com`
- **App ID**: `1:172157267811:web:df576a4e815f11adf1ad56`

## Required Steps to Enable Google Login

### 1. Get Service Account Credentials
1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Select your `eventflow-85373` project
3. Click the gear icon (⚙️) → **Project Settings**
4. Go to **Service accounts** tab
5. Click **"Generate new private key"**
6. Download the JSON file
7. From the JSON file, you need:
   - `project_id` (should be "eventflow-85373")
   - `private_key` (long string starting with "-----BEGIN PRIVATE KEY-----")
   - `client_email` (looks like "firebase-adminsdk-xxx@eventflow-85373.iam.gserviceaccount.com")

### 2. Enable Google Authentication
1. In Firebase Console, go to **Authentication**
2. Click **"Get started"** if not done already
3. Go to **Sign-in method** tab
4. Find **Google** and click it
5. Enable Google sign-in
6. Save the settings

### 3. Get Google OAuth Credentials
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select your `eventflow-85373` project
3. Go to **APIs & Services** → **Credentials**
4. Click **"Create Credentials"** → **"OAuth 2.0 Client ID"**
5. Choose **"Web application"**
6. Set name: "Enterprise RAG System"
7. Add **Authorized redirect URIs**:
   - `http://localhost:5000/auth/callback`
   - `https://your-replit-url.replit.app/auth/callback`
8. Click **Create**
9. Copy the **Client ID** and **Client Secret**

### 4. Environment Variables Needed

Once you have all the information, you'll need these 5 environment variables:

```
FIREBASE_PROJECT_ID=eventflow-85373
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n[YOUR_PRIVATE_KEY_HERE]\n-----END PRIVATE KEY-----\n"
FIREBASE_CLIENT_EMAIL=firebase-adminsdk-xxx@eventflow-85373.iam.gserviceaccount.com
GOOGLE_CLIENT_ID=[YOUR_CLIENT_ID].apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=[YOUR_CLIENT_SECRET]
```

### 5. Result
After setting these environment variables and restarting the app, users will see:
- **Google Login** tab (new!)
- **Standard Login** tab (existing)

Users can choose either method to authenticate.

## Quick Test
After setup, the authentication page will show both login options, and the first user to sign in with Google will automatically become an admin.