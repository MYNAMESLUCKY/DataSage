import os
import json
import logging
from typing import Dict, List, Any, Optional
import time

# AI/ML imports
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# OpenAI and HuggingFace
from openai import OpenAI

from .utils import setup_logging, cache_result
from .vector_store import VectorStoreManager

logger = setup_logging(__name__)

class RAGEngine:
    """
    Core RAG engine responsible for generating intelligent answers
    """
    
    def __init__(self):
        self.openai_client = None
        self.vector_store = None
        self.is_ready = False
        self.available_models = ["gpt-4o", "gpt-3.5-turbo", "huggingface"]
        
    def initialize(self):
        """Initialize the RAG engine with AI models"""
        try:
            logger.info("Initializing RAG Engine...")
            
            # Initialize OpenAI client
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                self.openai_client = OpenAI(api_key=openai_api_key)
                logger.info("OpenAI client initialized successfully")
            else:
                logger.warning("OpenAI API key not found. OpenAI models will not be available.")
            
            self.is_ready = True
            logger.info("RAG Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG Engine: {str(e)}")
            self.is_ready = False
    
    def update_vector_store(self, vector_store: VectorStoreManager):
        """Update the vector store reference"""
        self.vector_store = vector_store
        logger.info("Vector store updated in RAG engine")
    
    @cache_result(ttl=300)  # Cache for 5 minutes
    def generate_answer(
        self, 
        query: str, 
        relevant_docs: List[Document], 
        llm_model: str = "gpt-4o"
    ) -> Dict[str, Any]:
        """
        Generate an intelligent answer based on the query and relevant documents
        """
        try:
            logger.info(f"Generating answer using model: {llm_model}")
            
            # Prepare context from relevant documents
            context = self._prepare_context(relevant_docs)
            
            # Generate answer based on selected model
            if llm_model.startswith("gpt") and self.openai_client:
                result = self._generate_openai_answer(query, context, llm_model)
            else:
                # Fallback to a simpler approach
                result = self._generate_fallback_answer(query, context)
            
            # Extract sources
            sources = self._extract_sources(relevant_docs)
            
            return {
                'answer': result['answer'],
                'sources': sources,
                'confidence': result.get('confidence', 0.8),
                'model_used': llm_model
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                'answer': f"I encountered an error while processing your question: {str(e)}",
                'sources': [],
                'confidence': 0.0,
                'model_used': llm_model
            }
    
    def _prepare_context(self, relevant_docs: List[Document]) -> str:
        """Prepare context string from relevant documents"""
        context_parts = []
        
        for i, doc in enumerate(relevant_docs, 1):
            content = doc.page_content.strip()
            source = doc.metadata.get('source', f'Document {i}')
            
            context_parts.append(f"[Source {i}: {source}]\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _generate_openai_answer(self, query: str, context: str, model: str) -> Dict[str, Any]:
        """Generate answer using OpenAI models"""
        try:
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            
            prompt = f"""You are an intelligent AI assistant that provides accurate, helpful answers based on the provided context. 

Context Information:
{context}

Question: {query}

Instructions:
1. Provide a comprehensive and accurate answer based ONLY on the information in the context above
2. If the context doesn't contain enough information to fully answer the question, clearly state what information is missing
3. Include specific references to sources when possible
4. Be concise but thorough
5. If asked about topics not covered in the context, politely redirect to the available information

Please provide your answer in JSON format with the following structure:
{{"answer": "your detailed answer here", "confidence": confidence_score_between_0_and_1}}
"""

            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant specialized in analyzing documents and providing accurate answers."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            if content:
                result = json.loads(content)
            else:
                result = {"answer": "No response generated", "confidence": 0.0}
            
            return {
                'answer': result.get('answer', 'I was unable to generate a proper response.'),
                'confidence': result.get('confidence', 0.8)
            }
            
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            content = response.choices[0].message.content if 'response' in locals() else "Error parsing response"
            return {
                'answer': content,
                'confidence': 0.7
            }
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    def _generate_fallback_answer(self, query: str, context: str) -> Dict[str, Any]:
        """Generate a fallback answer when AI models are not available"""
        
        # Simple keyword-based approach
        query_lower = query.lower()
        context_lower = context.lower()
        
        # Find most relevant sentences
        sentences = context.split('.')
        relevant_sentences = []
        
        query_words = set(query_lower.split())
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words.intersection(sentence_words))
            
            if overlap > 1:  # At least 2 matching words
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            answer = "Based on the available information: " + " ".join(relevant_sentences[:3])
        else:
            answer = "I found some related information in the documents, but I cannot provide a specific answer to your question with the current processing capabilities. Please try rephrasing your question or ensure your data contains relevant information."
        
        return {
            'answer': answer,
            'confidence': 0.6
        }
    
    def _extract_sources(self, relevant_docs: List[Document]) -> List[str]:
        """Extract source information from documents"""
        sources = []
        seen_sources = set()
        
        for doc in relevant_docs:
            source = doc.metadata.get('source', 'Unknown Source')
            
            # Avoid duplicate sources
            if source not in seen_sources:
                sources.append(source)
                seen_sources.add(source)
        
        return sources[:5]  # Limit to top 5 sources
    
    def get_available_models(self) -> List[str]:
        """Get list of available AI models"""
        available = []
        
        if self.openai_client:
            available.extend(["gpt-4o", "gpt-3.5-turbo"])
        
        available.append("huggingface")  # Always available as fallback
        
        return available
    
    def estimate_context_relevance(self, query: str, context: str) -> float:
        """Estimate how relevant the context is to the query"""
        try:
            query_words = set(query.lower().split())
            context_words = set(context.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(query_words.intersection(context_words))
            union = len(query_words.union(context_words))
            
            if union == 0:
                return 0.0
            
            relevance = intersection / union
            return min(relevance * 2, 1.0)  # Scale up and cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating context relevance: {str(e)}")
            return 0.5  # Default moderate relevance
