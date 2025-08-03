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

logger = setup_logging(__name__)

class RAGEngine:
    """
    Core RAG engine responsible for generating intelligent answers
    """
    
    def __init__(self):
        self.openai_client = None
        self.vector_store = None
        self.is_ready = False
        self.available_models = ["sarvam-m", "deepseek-chat", "deepseek-coder", "moonshotai/kimi-k2:free", "gpt-4o", "gpt-3.5-turbo", "openai/gpt-4o", "openai/gpt-3.5-turbo", "anthropic/claude-3.5-sonnet", "meta-llama/llama-3.1-8b-instruct"]
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
            
            # Try SARVAM_API first (most reliable)
            if sarvam_api_key:
                try:
                    self.openai_client = OpenAI(
                        api_key=sarvam_api_key,
                        base_url="https://api.sarvam.ai/v1"
                    )
                    self.api_provider = "SARVAM"
                    self.default_model = "sarvam-m"
                    logger.info("SARVAM API client initialized successfully")
                except Exception as e:
                    logger.warning(f"SARVAM API failed: {e}, trying DeepSeek...")
            # Try DeepSeek native API second
            elif deepseek_api_key:
                try:
                    # First try as native DeepSeek API
                    self.openai_client = OpenAI(
                        api_key=deepseek_api_key,
                        base_url="https://api.deepseek.com"
                    )
                    self.api_provider = "DeepSeek"
                    self.default_model = "deepseek-chat"
                    logger.info("DeepSeek client initialized successfully")
                except Exception as e:
                    logger.warning(f"DeepSeek API failed: {e}, trying OpenRouter...")
                    # Fallback to OpenRouter if DeepSeek fails
                    if deepseek_api_key.startswith("sk-or-v1"):
                        self.openai_client = OpenAI(
                            api_key=deepseek_api_key,
                            base_url="https://openrouter.ai/api/v1"
                        )
                        self.api_provider = "OpenRouter"
                        self.default_model = "deepseek-chat"  # Use DeepSeek model on OpenRouter
                        logger.info("OpenRouter client initialized with DeepSeek model")
            elif openrouter_api_key:
                self.openai_client = OpenAI(
                    api_key=openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1"
                )
                self.api_provider = "OpenRouter"
                self.default_model = "moonshotai/kimi-k2:free"
                logger.info("OpenRouter client initialized successfully with Kimi model")
            elif openai_api_key:
                self.openai_client = OpenAI(api_key=openai_api_key)
                self.api_provider = "OpenAI"
                self.default_model = "gpt-4o"
                logger.info("OpenAI client initialized successfully")
            else:
                logger.warning("No API key found (OpenRouter, DeepSeek, or OpenAI). AI models will not be available.")
            
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
        """Generate answer using AI models with automatic fallback on rate limits"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if not self.openai_client:
                    raise Exception("No AI client available")
                
                # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
                # do not change this unless explicitly requested by the user
                
                prompt = f"""You are an expert knowledge assistant providing comprehensive, accurate answers.

CRITICAL INSTRUCTION: Provide your answer in clean, readable format WITHOUT embedding source citations mid-sentence. Sources will be listed separately.

Context Information:
{context}

Question: {query}

Response Guidelines:
1. Give a complete, well-structured answer based on the provided context
2. Start with the most direct answer to the question
3. Provide supporting details and explanations as needed
4. Do NOT include source references, citations, or [Source X] markers within your answer text
5. Write in clear, natural language as if explaining to someone who wants to understand the topic
6. If the context lacks information for a complete answer, acknowledge this at the end
7. Structure your response logically with proper paragraphs

Please provide your answer in JSON format:
{{"answer": "your clean, comprehensive answer without any source citations embedded", "confidence": confidence_score_between_0_and_1}}
"""

                # Adjust parameters based on model type
                request_params = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful AI assistant specialized in analyzing documents and providing accurate answers."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1000
                }
                
                # Only add response_format for models that support it (exclude certain models)
                if not any(model.startswith(prefix) for prefix in ["deepseek", "moonshotai/kimi", "sarvam"]):
                    request_params["response_format"] = {"type": "json_object"}

                response = self.openai_client.chat.completions.create(**request_params)
                
                content = response.choices[0].message.content
                if content:
                    # Try to parse JSON if available, otherwise use content directly
                    try:
                        result = json.loads(content)
                        return {
                            'answer': result.get('answer', content),
                            'confidence': result.get('confidence', 0.8)
                        }
                    except json.JSONDecodeError:
                        # If JSON parsing fails, use content directly
                        return {
                            'answer': content,
                            'confidence': 0.8
                        }
                else:
                    return {"answer": "No response generated", "confidence": 0.0}
                    
            except Exception as e:
                error_str = str(e)
                logger.error(f"AI API error (attempt {retry_count + 1}): {error_str}")
                
                # Check if this is a rate limit error
                if "429" in error_str or "rate limit" in error_str.lower():
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.info(f"Rate limit hit, trying fallback model (attempt {retry_count + 1})")
                        # Try fallback model if available
                        if self._try_fallback_model():
                            continue
                        else:
                            time.sleep(2 ** retry_count)  # Exponential backoff
                            continue
                    else:
                        logger.error("All retry attempts exhausted")
                        break
                else:
                    # Non-rate limit error, don't retry
                    break
        
        # If all retries failed, raise the last error
        raise Exception(f"Failed to generate AI answer after {max_retries} attempts: {error_str}")
    
    def _try_fallback_model(self) -> bool:
        """Try to switch to a fallback API provider"""
        try:
            # Try DeepSeek if currently using SARVAM and DeepSeek key is available
            if self.api_provider == "SARVAM":
                deepseek_api_key = os.getenv("DEEPSEEK_API")
                if deepseek_api_key:
                    self.openai_client = OpenAI(
                        api_key=deepseek_api_key,
                        base_url="https://api.deepseek.com"
                    )
                    self.api_provider = "DeepSeek"
                    self.default_model = "deepseek-chat"
                    logger.info("Switched to DeepSeek API as fallback")
                    return True
            
            # Try OpenAI if available
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                self.openai_client = OpenAI(api_key=openai_api_key)
                self.api_provider = "OpenAI"
                self.default_model = "gpt-4o"
                logger.info("Switched to OpenAI API as fallback")
                return True
                
            return False
        except Exception as e:
            logger.error(f"Failed to switch to fallback model: {e}")
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
