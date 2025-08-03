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

from ..utils.utils import setup_logging, cache_result
from .vector_store import VectorStoreManager
from .advanced_rate_limiter import global_rate_limiter, rate_limited_api_call

logger = setup_logging(__name__)

class RAGEngine:
    """
    Core RAG engine responsible for generating intelligent answers
    """
    
    def __init__(self):
        self.openai_client = None
        self.vector_store = None
        self.is_ready = False
        self.available_models = ["sarvam-m"]  # SARVAM only
        self.api_provider = "Unknown"
        
    def initialize(self):
        """Initialize the RAG engine with AI models"""
        try:
            logger.info("Initializing RAG Engine...")
            
            # Check available API keys with SARVAM_API as primary
            sarvam_api_key = os.getenv("SARVAM_API")
            deepseek_api_key = os.getenv("DEEPSEEK_API")
            openrouter_api_key = os.getenv("OPENROUTER_API")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            
            # Force SARVAM_API as primary (it's available)
            if sarvam_api_key:
                self.openai_client = OpenAI(
                    api_key=sarvam_api_key,
                    base_url="https://api.sarvam.ai/v1"
                )
                self.api_provider = "SARVAM"
                self.default_model = "sarvam-m"
                logger.info("SARVAM API client initialized successfully")
                
                # Test the connection
                try:
                    test_response = self.openai_client.chat.completions.create(
                        model="sarvam-m",
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=5
                    )
                    logger.info("SARVAM API connection verified")
                except Exception as e:
                    logger.warning(f"SARVAM API test failed: {e}, trying DeepSeek...")
                    sarvam_api_key = None  # Force fallback
            # Only allow SARVAM API - no fallbacks
            if not sarvam_api_key:
                raise Exception("SARVAM_API key is required. No other providers supported.")
            
            logger.info("SARVAM-only configuration enforced")
            
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
        llm_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an intelligent answer based on the query and relevant documents
        """
        try:
            # Use default model if none specified - ensure we always have a string
            if llm_model is None or llm_model == "":
                llm_model = getattr(self, 'default_model', 'sarvam-m')
            
            # Ensure llm_model is always a string at this point
            assert isinstance(llm_model, str), f"llm_model must be a string, got {type(llm_model)}"
            
            logger.info(f"Generating answer using model: {llm_model}")
            
            # Prepare context from relevant documents
            context = self._prepare_context(relevant_docs)
            
            # Generate answer based on available client and model
            if self.openai_client:
                result = self._generate_ai_answer(query, context, llm_model)
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
    
    def _generate_ai_answer(self, query: str, context: str, model: str) -> Dict[str, Any]:
        """Generate answer using AI models with intelligent rate limiting"""
        
        def make_api_call():
            """Inner function to make the actual API call"""
            if not self.openai_client:
                raise Exception("No AI client available")
                
            # Prepare prompt based on model type
            if model.startswith("sarvam"):
                prompt = f"""You are an expert knowledge assistant providing comprehensive, detailed, and specific answers with concrete examples and actionable information.

CRITICAL INSTRUCTION: Provide detailed, specific answers with examples, numbers, names, and concrete details. Avoid generic statements. Sources will be listed separately.

Context Information:
{context}

Question: {query}

Response Guidelines:
1. **BE SPECIFIC**: Include specific names, programs, schemes, amounts, dates, and concrete details from the context
2. **PROVIDE EXAMPLES**: Give real examples, case studies, or specific instances mentioned in the documents
3. **USE NUMBERS**: Include exact figures, percentages, amounts, or quantities when available
4. **STRUCTURE CLEARLY**: Use clear sections, bullet points, or numbered lists when appropriate
5. **START WITH KEY POINTS**: Lead with the most important and specific information
6. **AVOID GENERIC LANGUAGE**: Replace vague terms like "various schemes" with specific scheme names
7. **INCLUDE PRACTICAL DETAILS**: Add implementation details, eligibility criteria, or application processes if available
8. **NO SOURCE CITATIONS**: Do NOT include [Source X] or citation markers within your answer text

Please provide your answer directly as plain text with clear formatting and structure. Do NOT use JSON format.
"""
            else:
                prompt = f"""You are an expert knowledge assistant providing comprehensive, detailed, and specific answers with concrete examples and actionable information.

Context Information:
{context}

Question: {query}

Please provide your answer in JSON format:
{{"answer": "your detailed, specific answer with concrete examples and exact details", "confidence": confidence_score_between_0_and_1}}
"""

            # Configure request parameters
            request_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert research analyst who provides detailed, specific answers with concrete examples, exact figures, and actionable information. Always prioritize specificity over generality."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 1500
            }
            
            # Only add response_format for models that support it
            if not model.startswith("sarvam"):
                request_params["response_format"] = {"type": "json_object"}

            # Make the API call
            logger.info(f"Making SARVAM API call with model: {model}")
            
            try:
                response = self.openai_client.chat.completions.create(**request_params)
                logger.info(f"SARVAM API call completed successfully")
                
                # Validate response structure
                if not response or not hasattr(response, 'choices') or not response.choices:
                    raise Exception("Invalid API response structure")
                    
                content = response.choices[0].message.content
                if not content or not content.strip():
                    raise Exception("Empty response content")
                
                # Process response based on model type
                if model.startswith("sarvam"):
                    return {
                        'answer': content.strip(),
                        'confidence': 0.8,
                        'status': 'success',
                        'model_used': model,
                        'api_provider': self.api_provider
                    }
                else:
                    try:
                        result = json.loads(content)
                        return {
                            'answer': result.get('answer', content),
                            'confidence': result.get('confidence', 0.8),
                            'status': 'success',
                            'model_used': model,
                            'api_provider': self.api_provider
                        }
                    except json.JSONDecodeError:
                        return {
                            'answer': content.strip(),
                            'confidence': 0.8,
                            'status': 'success',
                            'model_used': model,
                            'api_provider': self.api_provider
                        }
                        
            except Exception as api_error:
                logger.error(f"SARVAM API call failed: {api_error}")
                return {
                    "answer": f"SARVAM API call failed: {str(api_error)}", 
                    "confidence": 0.0,
                    "status": "error",
                    "model_used": model,
                    "api_provider": self.api_provider
                }
        
        # Use advanced rate limiter for intelligent backoff
        try:
            return rate_limited_api_call(
                global_rate_limiter, 
                query, 
                make_api_call,
                max_retries=7  # Increased for complex queries
            )
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"API call failed: {e}")
            logger.warning(f"Falling back to knowledge base answer generation")
            
            # Check if it's a rate limit error and we have context
            if "rate limit" in str(e).lower() and context.strip():
                logger.info("Rate limit detected - using enhanced fallback with available knowledge")
                fallback_result = self._generate_enhanced_fallback_answer(query, context)
                fallback_result.update({
                    "model_used": f"{model} (fallback)",
                    "api_provider": f"{self.api_provider} (fallback)",
                    "status": "success_fallback",
                    "fallback_reason": "Rate limit exceeded - used knowledge base"
                })
                return fallback_result
            else:
                # For other errors, try basic fallback
                fallback_result = self._generate_fallback_answer(query, context)
                fallback_result.update({
                    "model_used": f"{model} (fallback)",
                    "api_provider": f"{self.api_provider} (fallback)", 
                    "status": "error_fallback",
                    "fallback_reason": f"API error: {str(e)}"
                })
                return fallback_result
    
    def _try_fallback_model(self) -> bool:
        """No fallback models allowed - SARVAM only"""
        logger.warning("Fallback models disabled - SARVAM API only")
        return False
    
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
    
    def _generate_enhanced_fallback_answer(self, query: str, context: str) -> Dict[str, Any]:
        """Generate an enhanced fallback answer using advanced text processing"""
        
        if not context.strip():
            return {
                'answer': "No relevant information is available in the knowledge base for your query.",
                'confidence': 0.1
            }
        
        # Enhanced processing for better answers
        query_lower = query.lower()
        
        # Split context into paragraphs and sentences
        paragraphs = [p.strip() for p in context.split('\n\n') if p.strip()]
        
        # Score paragraphs based on query relevance
        scored_paragraphs = []
        query_words = set(query_lower.split())
        
        for paragraph in paragraphs:
            if len(paragraph) < 50:  # Skip very short paragraphs
                continue
                
            para_words = set(paragraph.lower().split())
            overlap = len(query_words.intersection(para_words))
            
            # Calculate relevance score
            relevance_score = overlap / len(query_words) if query_words else 0
            
            if relevance_score > 0.1:  # At least 10% word overlap
                scored_paragraphs.append((paragraph, relevance_score))
        
        # Sort by relevance and take top paragraphs
        scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
        
        if scored_paragraphs:
            # Create comprehensive answer from top relevant paragraphs
            top_paragraphs = [p[0] for p in scored_paragraphs[:3]]
            
            answer = "Based on the information in the knowledge base:\n\n"
            for i, paragraph in enumerate(top_paragraphs, 1):
                # Clean up paragraph
                clean_para = paragraph.replace('[Source', '').replace(']', '').strip()
                if len(clean_para) > 200:
                    clean_para = clean_para[:200] + "..."
                answer += f"{i}. {clean_para}\n\n"
            
            answer += "Note: This answer was generated from the knowledge base due to API rate limits, but contains relevant information to address your question."
            
            confidence = min(0.85, scored_paragraphs[0][1] * 2)  # Cap at 85%
            
        else:
            # Fallback to simple sentence matching
            sentences = [s.strip() for s in context.split('.') if s.strip() and len(s.strip()) > 20]
            relevant_sentences = []
            
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                overlap = len(query_words.intersection(sentence_words))
                
                if overlap > 0:
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                answer = "From the available information: " + ". ".join(relevant_sentences[:5]) + "."
                confidence = 0.7
            else:
                answer = "The knowledge base contains information related to your topic, but I cannot provide a specific answer due to API limitations. Please try rephrasing your question or wait for API access to be restored."
                confidence = 0.3
        
        return {
            'answer': answer,
            'confidence': confidence
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
