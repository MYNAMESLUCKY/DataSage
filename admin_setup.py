#!/usr/bin/env python3
"""
Admin Setup Script for Enterprise RAG System
Use this script to promote users to admin role
"""

import sqlite3
import os
import sys

def promote_user_to_admin(email):
    """Promote a user to admin role"""
    if not os.path.exists('auth.db'):
        print("❌ Auth database does not exist. Users need to sign in first.")
        return False
    
    conn = sqlite3.connect('auth.db')
    cursor = conn.cursor()
    
    try:
        # Check if user exists
        cursor.execute('SELECT email, role FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        
        if user:
            current_email, current_role = user
            if current_role == 'admin':
                print(f"✅ {email} is already an admin")
            else:
                # Update user role to admin
                cursor.execute('UPDATE users SET role = ? WHERE email = ?', ('admin', email))
                conn.commit()
                print(f"✅ Successfully promoted {email} to admin role")
            return True
        else:
            print(f"❌ User {email} not found in database")
            print("Available users:")
            cursor.execute('SELECT email, role FROM users ORDER BY created_at DESC')
            users = cursor.fetchall()
            for user_email, role in users:
                print(f"  • {user_email} ({role})")
            return False
            
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False
    finally:
        conn.close()

def list_users():
    """List all users in the database"""
    if not os.path.exists('auth.db'):
        print("❌ Auth database does not exist")
        return
    
    conn = sqlite3.connect('auth.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('SELECT email, role, created_at FROM users ORDER BY created_at DESC')
        users = cursor.fetchall()
        
        if users:
            print("📋 All Users:")
            for email, role, created_at in users:
                role_icon = "👑" if role == "admin" else "👤"
                print(f"  {role_icon} {email} - {role} (created: {created_at})")
        else:
            print("No users found in database")
            
    except Exception as e:
        print(f"❌ Database error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    print("🔧 Enterprise RAG Admin Setup")
    print("=" * 40)
    
    if len(sys.argv) == 2:
        if sys.argv[1] == "list":
            list_users()
        else:
            email = sys.argv[1]
            promote_user_to_admin(email)
    else:
        print("Usage:")
        print(f"  python {sys.argv[0]} <email>     # Promote user to admin")
        print(f"  python {sys.argv[0]} list       # List all users")
        print()
        print("Example:")
        print(f"  python {sys.argv[0]} your-email@gmail.com")
        print()
        list_users()