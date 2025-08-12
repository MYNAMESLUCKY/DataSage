"""
Advanced Query Processing for Enterprise RAG System
Implements query rewriting, expansion, and routing for improved retrieval
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from openai import OpenAI
from ..utils.utils import setup_logging

logger = setup_logging(__name__)

class AdvancedQueryProcessor:
    """
    Advanced query processing with rewriting, expansion, and routing capabilities
    Based on 2025 enterprise RAG best practices for +4-6 points NDCG improvement
    """
    
    def __init__(self, rag_engine=None):
        self.rag_engine = rag_engine
        self.client = None
        self.model = "sarvam-m"  # Default to SARVAM
        self.executor = ThreadPoolExecutor(max_workers=3)
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client for query processing"""
        try:
            # Use same client as RAG engine if available
            if self.rag_engine and hasattr(self.rag_engine, 'openai_client'):
                self.client = self.rag_engine.openai_client
                self.model = getattr(self.rag_engine, 'default_model', 'sarvam-m')
            
                logger.info(f"Using RAG engine client with model: {self.model}")
                return
            
            # Initialize our own client - SARVAM only
            sarvam_api_key = os.getenv("SARVAM_API")
            if not sarvam_api_key:
                raise Exception("SARVAM_API key required for query processing")
            self.model = getattr(self)
            
            self.client = OpenAI(
                api_key=sarvam_api_key,
                base_url="https://api.sarvam.ai/v1"
            )
            self.model = "sarvam-m"
            logger.info("Query processor initialized with SARVAM API (SARVAM-only mode)")
                
        except Exception as e:
            logger.error(f"Failed to initialize query processor: {e}")
    
    def generate_query_rewrites(self, original_query: str, num_rewrites: int = 3) -> List[str]:
        """
        Generate multiple query rewrites for improved retrieval
        Azure AI Search reports 147ms for 10 rewrites, we target 5 for balance
        """
        if not self.client:
            logger.warning("No client available for query rewriting")
            return [original_query]
        
        try:
            start_time = time.time()
            
            prompt = f"""You are an expert query rewriter for enterprise document search. 
Given a user query, generate {num_rewrites} alternative phrasings that would help retrieve the same information from different perspectives.

Original Query: "{original_query}"

Generate {num_rewrites} rewrites that:
1. Use different terminology and synonyms
2. Vary the structure and approach
3. Consider technical vs. business language
4. Include relevant domain-specific terms
5. Maintain the original intent

Respond with JSON format:
{{"rewrites": ["rewrite1", "rewrite2", "rewrite3", "rewrite4", "rewrite5"]}}"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=400  # Reduced for rate limiting optimization
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                data = json.loads(result)
                rewrites = data.get("rewrites", [])
                
                # Add original query to the list
                all_queries = [original_query] + rewrites
                
                elapsed_time = time.time() - start_time
                logger.info(f"Generated {len(rewrites)} query rewrites in {elapsed_time:.2f}s")
                
                return all_queries[:num_rewrites + 1]  # Include original + rewrites
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse query rewrite JSON, using original query")
                return [original_query]
                
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return [original_query]
    
    def expand_query_with_context(self, query: str, context_docs: List[Any] = None) -> str:
        """
        Expand query with contextual information from relevant documents
        """
        if not self.client or not context_docs:
            return query
        
        try:
            # Extract key terms from context documents
            context_text = ""
            if context_docs:
                context_snippets = []
                for doc in context_docs[:3]:  # Use top 3 documents
                    content = getattr(doc, 'page_content', str(doc))
                    context_snippets.append(content[:200])  # First 200 chars
                context_text = " | ".join(context_snippets)
            
            prompt = f"""Based on the following context, expand this query to be more specific and comprehensive:

Original Query: "{query}"

Context: {context_text}

Provide an expanded query that:
1. Includes relevant technical terms from the context
2. Adds specific details that would improve search
3. Maintains the original intent
4. Stays concise (under 50 words)

Expanded Query:"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )
            
            expanded_query = response.choices[0].message.content.strip()
            logger.info(f"Expanded query: {query} -> {expanded_query}")
            
            return expanded_query
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return query
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """
        Determine optimal routing strategy for the query
        Routes to: vector_search, web_search, hybrid, or knowledge_graph
        """
        try:
            if not self.client:
                return {"strategy": "hybrid", "confidence": 0.5, "reasoning": "Default routing"}
            
            prompt = f"""Analyze this query and determine the best search strategy:

Query: "{query}"

Classify into one of these categories:
1. "internal_knowledge" - Company/domain-specific information, policies, procedures
2. "factual_lookup" - Current facts, statistics, recent information 
3. "technical_deep_dive" - Complex technical concepts requiring multiple sources
4. "comparative_analysis" - Comparing multiple items, requires comprehensive data
5. "real_time_data" - Current events, live data, recent news

Respond with JSON:
{{"strategy": "category", "confidence": 0.8, "reasoning": "explanation", "web_search_needed": true/false}}"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=200
            )
            
            result = response.choices[0].message.content.strip()
            
            try:
                routing_data = json.loads(result)
                logger.info(f"Query routing: {routing_data.get('strategy')} (confidence: {routing_data.get('confidence')})")
                return routing_data
            except json.JSONDecodeError:
                logger.warning("Failed to parse routing JSON")
                return {"strategy": "hybrid", "confidence": 0.5, "reasoning": "Parsing failed"}
                
        except Exception as e:
            logger.error(f"Query routing failed: {e}")
            return {"strategy": "hybrid", "confidence": 0.5, "reasoning": f"Error: {e}"}
    
    def break_down_complex_query(self, query: str) -> List[str]:
        """
        Break down complex queries into sub-queries for comprehensive retrieval
        Essential for comparative analysis and multi-faceted questions
        """
        if not self.client:
            return [query]
        
        try:
            prompt = f"""Analyze this query and break it down into 2-4 focused sub-queries if it's complex:

Query: "{query}"

If this is a simple, single-concept query, respond with: {{"sub_queries": ["{query}"]}}

If it's complex, break it into specific sub-queries that together would answer the original question.
Examples of complex queries:
- Comparative questions (X vs Y)
- Multi-part questions (How does X work and what are its benefits?)
- Questions requiring different types of information

Respond with JSON:
{{"sub_queries": ["sub_query_1", "sub_query_2", "sub_query_3"], "is_complex": true/false}}"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            
            result = response.choices[0].message.content.strip()
            
            try:
                data = json.loads(result)
                sub_queries = data.get("sub_queries", [query])
                is_complex = data.get("is_complex", False)
                
                if is_complex:
                    logger.info(f"Complex query broken into {len(sub_queries)} sub-queries")
                else:
                    logger.info("Query identified as simple, no breakdown needed")
                
                return sub_queries
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse sub-query JSON")
                return [query]
                
        except Exception as e:
            logger.error(f"Query breakdown failed: {e}")
            return [query]
    
    def process_query_comprehensive(self, query: str) -> Dict[str, Any]:
        """
        Comprehensive query processing combining all techniques
        Returns processed queries and routing information
        """
        start_time = time.time()
        
        try:
            # Step 1: Route the query
            routing_info = self.route_query(query)
            
            # Step 2: Break down if complex
            sub_queries = self.break_down_complex_query(query)
            
            # Step 3: Generate rewrites for main query and sub-queries
            all_rewrites = {}
            
            # Process main query
            main_rewrites = self.generate_query_rewrites(query, num_rewrites=3)
            all_rewrites['main'] = main_rewrites
            
            # Process sub-queries if different from main
            if len(sub_queries) > 1:
                for i, sub_query in enumerate(sub_queries):
                    if sub_query != query:  # Don't rewrite if same as main
                        sub_rewrites = self.generate_query_rewrites(sub_query, num_rewrites=2)
                        all_rewrites[f'sub_{i}'] = sub_rewrites
            
            processing_time = time.time() - start_time
            
            result = {
                'original_query': query,
                'routing': routing_info,
                'sub_queries': sub_queries,
                'query_rewrites': all_rewrites,
                'processing_time': processing_time,
                'total_queries': sum(len(rewrites) for rewrites in all_rewrites.values())
            }
            
            logger.info(f"Comprehensive query processing completed in {processing_time:.2f}s - {result['total_queries']} total queries generated")
            
            return result
            
        except Exception as e:
            logger.error(f"Comprehensive query processing failed: {e}")
            return {
                'original_query': query,
                'routing': {"strategy": "hybrid", "confidence": 0.5},
                'sub_queries': [query],
                'query_rewrites': {'main': [query]},
                'processing_time': time.time() - start_time,
                'total_queries': 1
            }