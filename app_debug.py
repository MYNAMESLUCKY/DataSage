import streamlit as st
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'components'))

def main():
    try:
        st.set_page_config(
            page_title="RAG System Debug",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🧠 Enterprise RAG System")
        st.write("Debug version to test basic functionality")
        
        # Test imports
        try:
            from backend.api import RAGSystemAPI
            st.success("✅ Backend API imported successfully")
            
            api = RAGSystemAPI()
            st.success("✅ API initialized successfully")
            
            # Test stats
            stats = api.get_system_stats()
            st.write(f"Documents loaded: {stats.get('total_documents', 0)}")
            
        except Exception as e:
            st.error(f"❌ Backend error: {str(e)}")
        
        # Test UI components
        try:
            from components.ui_components import UIComponents
            st.success("✅ UI Components imported successfully")
            
            ui = UIComponents()
            st.success("✅ UI Components initialized successfully")
            
        except Exception as e:
            st.error(f"❌ UI Components error: {str(e)}")
        
        # Test query functionality
        st.subheader("Test Query")
        query = st.text_input("Enter a test query:", value="What is machine learning?")
        
        if st.button("Test Query"):
            try:
                result = api.query(query, llm_model="moonshotai/kimi-k2:free")
                st.success("✅ Query successful!")
                st.write("Answer:", result.get('answer', 'No answer'))
                st.write("Sources:", len(result.get('sources', [])))
                
                # Add copy functionality for the answer
                if result.get('answer'):
                    st.text_area(
                        "Copy this answer:",
                        value=result.get('answer', ''),
                        height=200,
                        help="Select all (Ctrl+A) and copy (Ctrl+C)"
                    )
            except Exception as e:
                st.error(f"❌ Query failed: {str(e)}")
                
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()