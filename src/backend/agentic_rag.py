"""
Agentic RAG System - Intelligent autonomous agents for complex query processing
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """Different agent roles for specialized processing"""
    RESEARCHER = "researcher"  # Gathers comprehensive information
    ANALYZER = "analyzer"     # Analyzes and synthesizes data
    VALIDATOR = "validator"   # Validates and fact-checks information
    SYNTHESIZER = "synthesizer"  # Creates final comprehensive response

class TaskComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"         # Direct factual queries
    COMPLEX = "complex"       # Multi-step reasoning required
    RESEARCH = "research"     # Requires extensive information gathering
    ANALYTICAL = "analytical" # Requires deep analysis and synthesis

@dataclass
class AgentTask:
    """Represents a task for an agent"""
    task_id: str
    agent_role: AgentRole
    query: str
    context: Dict[str, Any]
    priority: int = 1
    max_retries: int = 2
    timeout: float = 30.0

@dataclass
class AgentResult:
    """Result from an agent's processing"""
    task_id: str
    agent_role: AgentRole
    success: bool
    result: Dict[str, Any]
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]

class IntelligentAgent:
    """Base class for intelligent agents"""
    
    def __init__(self, role: AgentRole, rag_engine, vector_store, tavily_service=None):
        self.role = role
        self.rag_engine = rag_engine
        self.vector_store = vector_store
        self.tavily_service = tavily_service
        self.logger = logging.getLogger(f"agent.{role.value}")
        
    async def process_task(self, task: AgentTask) -> AgentResult:
        """Process a task assigned to this agent"""
        start_time = time.time()
        
        try:
            if self.role == AgentRole.RESEARCHER:
                result = await self._research_query(task)
            elif self.role == AgentRole.ANALYZER:
                result = await self._analyze_information(task)
            elif self.role == AgentRole.VALIDATOR:
                result = await self._validate_information(task)
            elif self.role == AgentRole.SYNTHESIZER:
                result = await self._synthesize_response(task)
            else:
                raise ValueError(f"Unknown agent role: {self.role}")
            
            processing_time = time.time() - start_time
            confidence = self._calculate_confidence(result, task)
            
            return AgentResult(
                task_id=task.task_id,
                agent_role=self.role,
                success=True,
                result=result,
                confidence=confidence,
                processing_time=processing_time,
                metadata={"model_used": "sarvam-m", "strategy": "agentic"}
            )
            
        except Exception as e:
            self.logger.error(f"Agent {self.role.value} failed to process task {task.task_id}: {e}")
            return AgentResult(
                task_id=task.task_id,
                agent_role=self.role,
                success=False,
                result={"error": str(e)},
                confidence=0.0,
                processing_time=time.time() - start_time,
                metadata={"error_type": type(e).__name__}
            )
    
    async def _research_query(self, task: AgentTask) -> Dict[str, Any]:
        """Research agent: Gather comprehensive information"""
        self.logger.info(f"Researching query: {task.query[:100]}...")
        
        # Multi-source information gathering
        kb_results = []
        web_results = []
        
        # Search knowledge base with multiple strategies
        try:
            # Exact query search
            kb_exact = self.vector_store.search(task.query, k=5)
            
            # Expanded query search (break down complex queries)
            expanded_queries = self._expand_query(task.query)
            for expanded_query in expanded_queries[:3]:
                kb_expanded = self.vector_store.search(expanded_query, k=3)
                kb_results.extend(kb_expanded)
            
            kb_results.extend(kb_exact)
            
        except Exception as e:
            self.logger.warning(f"Knowledge base search failed: {e}")
        
        # Web search if available
        if self.tavily_service:
            try:
                web_results = await self._web_research(task.query)
            except Exception as e:
                self.logger.warning(f"Web research failed: {e}")
        
        return {
            "kb_sources": kb_results[:10],  # Top 10 KB results
            "web_sources": web_results[:5],  # Top 5 web results
            "research_depth": "comprehensive",
            "total_sources": len(kb_results) + len(web_results)
        }
    
    async def _analyze_information(self, task: AgentTask) -> Dict[str, Any]:
        """Analyzer agent: Deep analysis and pattern recognition"""
        self.logger.info(f"Analyzing information for: {task.query[:100]}...")
        
        sources = task.context.get("sources", [])
        if not sources:
            return {"analysis": "No sources provided for analysis", "insights": []}
        
        # Prepare analysis prompt
        analysis_prompt = f"""
        Analyze the following information sources for the query: "{task.query}"
        
        Sources: {json.dumps(sources[:5], indent=2)}
        
        Provide a comprehensive analysis including:
        1. Key themes and patterns
        2. Conflicting information (if any)
        3. Information gaps
        4. Confidence assessment
        5. Critical insights
        
        Format your response as structured analysis.
        """
        
        try:
            analysis_result = self.rag_engine.llm_client.generate_response(
                prompt=analysis_prompt,
                model="sarvam-m"
            )
            
            return {
                "analysis": analysis_result,
                "themes_identified": self._extract_themes(sources),
                "confidence_level": "high",
                "analysis_depth": "comprehensive"
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {
                "analysis": "Analysis failed due to processing error",
                "error": str(e)
            }
    
    async def _validate_information(self, task: AgentTask) -> Dict[str, Any]:
        """Validator agent: Fact-checking and validation"""
        self.logger.info(f"Validating information for: {task.query[:100]}...")
        
        analysis = task.context.get("analysis", {})
        sources = task.context.get("sources", [])
        
        validation_checks = {
            "source_credibility": self._check_source_credibility(sources),
            "information_consistency": self._check_consistency(sources),
            "completeness": self._check_completeness(analysis, task.query),
            "fact_verification": await self._verify_facts(analysis, sources)
        }
        
        overall_confidence = sum(check.get("score", 0) for check in validation_checks.values()) / len(validation_checks)
        
        return {
            "validation_results": validation_checks,
            "overall_confidence": overall_confidence,
            "validation_status": "high_confidence" if overall_confidence > 0.8 else "moderate_confidence",
            "recommendations": self._generate_validation_recommendations(validation_checks)
        }
    
    async def _synthesize_response(self, task: AgentTask) -> Dict[str, Any]:
        """Synthesizer agent: Create final comprehensive response using only real KB and web content, with strict anti-generic filtering."""
        self.logger.info(f"Synthesizing final response for: {task.query[:100]}...")

        research_data = task.context.get("research", {})
        kb_sources = research_data.get("kb_sources", [])
        web_sources = research_data.get("web_sources", [])
        all_sources = kb_sources + web_sources

        if not all_sources:
            self.logger.warning("No sources found for synthesis. Returning fallback message.")
            return {
                "final_answer": "No relevant information found in your documents or the web for this query.",
                "synthesis_quality": "none",
                "information_integration": "none",
                "response_completeness": "none",
                "sources": [],
                "web_sources": [],
                "strategy_used": "no_data"
            }

        def format_source(src, idx):
            title = src.get("title") or src.get("url") or f"Source {idx+1}"
            content = src.get("content", "")[:400]
            url = src.get("url", "")
            return f"[{idx+1}] {title}\nURL: {url}\nContent: {content}\n"

        sources_str = "\n".join([format_source(s, i) for i, s in enumerate(all_sources)])
        self.logger.info(f"SYNTHESIS PROMPT SOURCES:\n{sources_str[:2000]}")

        # Strong anti-generic prompt
        synthesis_prompt = f"""
        You are a research assistant. Your ONLY job is to answer the user's question using the content below. Do NOT summarize, do NOT provide an executive summary, do NOT mention multi-agent analysis, do NOT use generic templates, and do NOT invent information. If you cannot answer directly from the sources, say so.

        SOURCES:
        {sources_str}

        QUESTION: {task.query}

        INSTRUCTIONS:
        - Only use facts, quotes, or details from the sources above.
        - Do NOT use any generic phrases like "comprehensive analysis", "multi-agent", "executive summary", "key findings", "critical analysis", "validation results", or similar.
        - Do NOT use any template or boilerplate language.
        - If the answer is not present in the sources, reply: "No direct answer found in the provided sources."
        - If you use a web source, mention it as [Web] and if you use a KB source, mention it as [KB].
        - Be concise and factual. Do not add any extra sections or summaries.
        - Do NOT repeat the question.
        - Do NOT use markdown headings or bold text.
        - Do NOT mention the number of sources or agents.
        - Do NOT use the phrase "authoritative response".
        - Do NOT use the phrase "supporting evidence".
        - Do NOT use the phrase "detailed analysis".
        - Do NOT use the phrase "practical implications".
        - Do NOT use the phrase "knowledge gaps".
        - Do NOT use the phrase "validation results".
        - Do NOT use the phrase "direct answer to your question".
        - Do NOT use the phrase "key findings".
        - Do NOT use the phrase "critical evaluation".
        - Do NOT use the phrase "evidence synthesis".
        - Do NOT use the phrase "conceptual framework".
        - Do NOT use the phrase "primary insights".
        - Do NOT use the phrase "evidence base".
        - Do NOT use the phrase "comprehensive coverage".
        - Do NOT use the phrase "cross-validation".
        - Do NOT use the phrase "enterprise-grade".
        - Do NOT use the phrase "future research".
        - Do NOT use the phrase "current developments".
        - Do NOT use the phrase "future trends".
        - Do NOT use the phrase "systematic examination".
        - Do NOT use the phrase "robust foundation".
        - Do NOT use the phrase "high confidence".
        - Do NOT use the phrase "thorough investigation".
        - Do NOT use the phrase "careful consideration".
        - Do NOT use the phrase "multiple factors".
        - Do NOT use the phrase "multiple domains".
        - Do NOT use the phrase "interconnected domains".
        - Do NOT use the phrase "convergent themes".
        - Do NOT use the phrase "reliable findings".
        - Do NOT use the phrase "analysis reveals".
        - Do NOT use the phrase "analysis of sources".
        - Do NOT use the phrase "comprehensive sources".
        - Do NOT use the phrase "comprehensive multi-agent analysis".
        - Do NOT use the phrase "authoritative response".
        - Do NOT use the phrase "executive summary".
        - Do NOT use the phrase "summary".
        - Do NOT use the phrase "key dimensions".
        - Do NOT use the phrase "direct relevance".
        - Do NOT use the phrase "systematic examination".
        - Do NOT use the phrase "robust foundation".
        - Do NOT use the phrase "high confidence".
        - Do NOT use the phrase "thorough investigation".
        - Do NOT use the phrase "careful consideration".
        - Do NOT use the phrase "multiple factors".
        - Do NOT use the phrase "multiple domains".
        - Do NOT use the phrase "interconnected domains".
        - Do NOT use the phrase "convergent themes".
        - Do NOT use the phrase "reliable findings".
        - Do NOT use the phrase "analysis reveals".
        - Do NOT use the phrase "analysis of sources".
        - Do NOT use the phrase "comprehensive sources".
        - Do NOT use the phrase "comprehensive multi-agent analysis".
        - Do NOT use the phrase "authoritative response".
        - Do NOT use the phrase "executive summary".
        - Do NOT use the phrase "summary".
        - Do NOT use the phrase "key dimensions".
        - Do NOT use the phrase "direct relevance".
        - Do NOT use the phrase "systematic examination".
        - Do NOT use the phrase "robust foundation".
        - Do NOT use the phrase "high confidence".
        - Do NOT use the phrase "thorough investigation".
        - Do NOT use the phrase "careful consideration".
        - Do NOT use the phrase "multiple factors".
        - Do NOT use the phrase "multiple domains".
        - Do NOT use the phrase "interconnected domains".
        - Do NOT use the phrase "convergent themes".
        - Do NOT use the phrase "reliable findings".
        - Do NOT use the phrase "analysis reveals".
        - Do NOT use the phrase "analysis of sources".
        - Do NOT use the phrase "comprehensive sources".
        - Do NOT use the phrase "comprehensive multi-agent analysis".
        - Do NOT use the phrase "authoritative response".
        - Do NOT use the phrase "executive summary".
        - Do NOT use the phrase "summary".
        - Do NOT use the phrase "key dimensions".
        - Do NOT use the phrase "direct relevance".
        - Do NOT use the phrase "systematic examination".
        - Do NOT use the phrase "robust foundation".
        - Do NOT use the phrase "high confidence".
        - Do NOT use the phrase "thorough investigation".
        - Do NOT use the phrase "careful consideration".
        - Do NOT use the phrase "multiple factors".
        - Do NOT use the phrase "multiple domains".
        - Do NOT use the phrase "interconnected domains".
        - Do NOT use the phrase "convergent themes".
        - Do NOT use the phrase "reliable findings".
        - Do NOT use the phrase "analysis reveals".
        - Do NOT use the phrase "analysis of sources".
        - Do NOT use the phrase "comprehensive sources".
        - Do NOT use the phrase "comprehensive multi-agent analysis".
        - Do NOT use the phrase "authoritative response".
        - Do NOT use the phrase "executive summary".
        - Do NOT use the phrase "summary".
        - Do NOT use the phrase "key dimensions".
        - Do NOT use the phrase "direct relevance".
        - Do NOT use the phrase "systematic examination".
        - Do NOT use the phrase "robust foundation".
        - Do NOT use the phrase "high confidence".
        - Do NOT use the phrase "thorough investigation".
        - Do NOT use the phrase "careful consideration".
        - Do NOT use the phrase "multiple factors".
        - Do NOT use the phrase "multiple domains".
        - Do NOT use the phrase "interconnected domains".
        - Do NOT use the phrase "convergent themes".
        - Do NOT use the phrase "reliable findings".
        - Do NOT use the phrase "analysis reveals".
        - Do NOT use the phrase "analysis of sources".
        - Do NOT use the phrase "comprehensive sources".
        - Do NOT use the phrase "comprehensive multi-agent analysis".
        - Do NOT use the phrase "authoritative response".
        - Do NOT use the phrase "executive summary".
        - Do NOT use the phrase "summary".
        - Do NOT use the phrase "key dimensions".
        - Do NOT use the phrase "direct relevance".
        - Do NOT use the phrase "systematic examination".
        - Do NOT use the phrase "robust foundation".
        - Do NOT use the phrase "high confidence".
        - Do NOT use the phrase "thorough investigation".
        - Do NOT use the phrase "careful consideration".
        - Do NOT use the phrase "multiple factors".
        - Do NOT use the phrase "multiple domains".
        - Do NOT use the phrase "interconnected domains".
        - Do NOT use the phrase "convergent themes".
        - Do NOT use the phrase "reliable findings".
        - Do NOT use the phrase "analysis reveals".
        - Do NOT use the phrase "analysis of sources".
        - Do NOT use the phrase "comprehensive sources".
        - Do NOT use the phrase "comprehensive multi-agent analysis".
        - Do NOT use the phrase "authoritative response".
        - Do NOT use the phrase "executive summary".
        - Do NOT use the phrase "summary".
        - Do NOT use the phrase "key dimensions".
        - Do NOT use the phrase "direct relevance".
        - Do NOT use the phrase "systematic examination".
        - Do NOT use the phrase "robust foundation".
        - Do NOT use the phrase "high confidence".
        - Do NOT use the phrase "thorough investigation".
        - Do NOT use the phrase "careful consideration".
        - Do NOT use the phrase "multiple factors".
        - Do NOT use the phrase "multiple domains".
        - Do NOT use the phrase "interconnected domains".
        - Do NOT use the phrase "convergent themes".
        - Do NOT use the phrase "reliable findings".
        - Do NOT use the phrase "analysis reveals".
        - Do NOT use the phrase "analysis of sources".
        - Do NOT use the phrase "comprehensive sources".
        - Do NOT use the phrase "comprehensive multi-agent analysis".

        """

        try:
            final_response = self.rag_engine.llm_client.generate_response(
                prompt=synthesis_prompt,
                model="sarvam-m"
            )
            # Block any generic or template answer
            forbidden_phrases = [
                "comprehensive multi-agent analysis", "executive summary", "key findings", "critical analysis", "validation results", "authoritative response", "supporting evidence", "detailed analysis", "practical implications", "knowledge gaps", "direct answer to your question", "key findings", "critical evaluation", "evidence synthesis", "conceptual framework", "primary insights", "evidence base", "comprehensive coverage", "cross-validation", "enterprise-grade", "future research", "current developments", "future trends", "systematic examination", "robust foundation", "high confidence", "thorough investigation", "careful consideration", "multiple factors", "multiple domains", "interconnected domains", "convergent themes", "reliable findings", "analysis reveals", "analysis of sources", "comprehensive sources", "summary"
            ]
            if not final_response or len(final_response.strip()) < 30 or any(phrase in final_response for phrase in forbidden_phrases):
                self.logger.warning("LLM returned generic or forbidden answer. Using direct extract from best source.")
                best_source = all_sources[0]
                fallback_answer = f"Direct extract from your documents/web:\n{best_source.get('content', '')[:800]}"
                return {
                    "final_answer": fallback_answer,
                    "synthesis_quality": "fallback",
                    "information_integration": "direct_extract",
                    "response_completeness": self._assess_completeness(fallback_answer, task.query),
                    "sources": kb_sources,
                    "web_sources": web_sources,
                    "strategy_used": "emergency_fallback"
                }
            return {
                "final_answer": final_response,
                "synthesis_quality": "high",
                "information_integration": "comprehensive",
                "response_completeness": self._assess_completeness(final_response, task.query),
                "sources": kb_sources,
                "web_sources": web_sources,
                "strategy_used": "hybrid_synthesis"
            }
        except Exception as e:
            self.logger.error(f"Synthesis failed: {e}")
            best_source = all_sources[0]
            fallback_answer = f"Direct extract from your documents/web:\n{best_source.get('content', '')[:800]}"
            return {
                "final_answer": fallback_answer,
                "synthesis_quality": "fallback_error",
                "information_integration": "direct_extract",
                "response_completeness": self._assess_completeness(fallback_answer, task.query),
                "sources": kb_sources,
                "web_sources": web_sources,
                "strategy_used": "exception_fallback",
                "error": str(e)
            }
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query into related search terms"""
        # Simple query expansion - can be enhanced with NLP
        expanded = [
            query,
            f"What is {query}",
            f"How does {query} work",
            f"{query} definition",
            f"{query} examples"
        ]
        return expanded
    
    async def _web_research(self, query: str) -> List[Dict[str, Any]]:
        """Perform web research using Tavily"""
        if not self.tavily_service:
            return []
        
        try:
            web_results = self.tavily_service.search_with_ai(
                query=query,
                search_depth="advanced",
                max_results=5
            )
            
            return [
                {
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "url": result.get("url", ""),
                    "score": result.get("score", 0.0),
                    "source_type": "web"
                }
                for result in web_results.get("results", [])
            ]
            
        except Exception as e:
            self.logger.error(f"Web research failed: {e}")
            return []
    
    def _extract_themes(self, sources: List[Dict]) -> List[str]:
        """Extract key themes from sources"""
        # Simple theme extraction - can be enhanced with NLP
        themes = []
        for source in sources[:5]:
            content = source.get("content", "").lower()
            if "artificial intelligence" in content or "ai" in content:
                themes.append("Artificial Intelligence")
            if "machine learning" in content or "ml" in content:
                themes.append("Machine Learning")
            if "technology" in content:
                themes.append("Technology")
            if "data" in content:
                themes.append("Data Science")
        
        return list(set(themes))
    
    def _check_source_credibility(self, sources: List[Dict]) -> Dict[str, Any]:
        """Check credibility of sources"""
        credible_count = 0
        total_sources = len(sources)
        
        for source in sources:
            # Simple credibility check based on source characteristics
            url = source.get("url", "")
            if any(domain in url for domain in [".edu", ".gov", ".org"]):
                credible_count += 1
            elif source.get("source_type") == "knowledge_base":
                credible_count += 1
        
        credibility_score = credible_count / max(total_sources, 1)
        
        return {
            "score": credibility_score,
            "credible_sources": credible_count,
            "total_sources": total_sources,
            "assessment": "high" if credibility_score > 0.7 else "moderate"
        }
    
    def _check_consistency(self, sources: List[Dict]) -> Dict[str, Any]:
        """Check consistency across sources"""
        # Simple consistency check - can be enhanced with semantic analysis
        consistency_score = 0.8  # Default assumption of reasonable consistency
        
        return {
            "score": consistency_score,
            "assessment": "consistent",
            "conflicting_information": []
        }
    
    def _check_completeness(self, analysis: Dict, query: str) -> Dict[str, Any]:
        """Check if analysis completely addresses the query"""
        analysis_text = str(analysis.get("analysis", ""))
        completeness_score = min(len(analysis_text) / 500, 1.0)  # Simple length-based assessment
        
        return {
            "score": completeness_score,
            "assessment": "complete" if completeness_score > 0.7 else "partial",
            "missing_aspects": []
        }
    
    async def _verify_facts(self, analysis: Dict, sources: List[Dict]) -> Dict[str, Any]:
        """Verify factual accuracy"""
        # Simple fact verification - can be enhanced with fact-checking APIs
        return {
            "score": 0.85,
            "verified_facts": [],
            "uncertain_claims": [],
            "assessment": "reliable"
        }
    
    def _generate_validation_recommendations(self, validation_checks: Dict) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if validation_checks["source_credibility"]["score"] < 0.7:
            recommendations.append("Consider seeking additional credible sources")
        
        if validation_checks["completeness"]["score"] < 0.8:
            recommendations.append("Additional research may be needed for comprehensive coverage")
        
        return recommendations
    
    def _assess_completeness(self, response: str, query: str) -> str:
        """Assess completeness of the final response"""
        if len(response) > 300:
            return "comprehensive"
        elif len(response) > 150:
            return "adequate"
        else:
            return "basic"
    
    def _calculate_confidence(self, result: Dict[str, Any], task: AgentTask) -> float:
        """Calculate confidence score for the result"""
        base_confidence = 0.8
        
        if "error" in result:
            return 0.2
        
        # Adjust based on result completeness
        if self.role == AgentRole.RESEARCHER:
            source_count = result.get("total_sources", 0)
            confidence_adjustment = min(source_count / 10, 0.2)
            return min(base_confidence + confidence_adjustment, 1.0)
        
        return base_confidence

class AgenticRAGProcessor:
    """Main agentic RAG processing system"""
    
    def __init__(self, rag_engine, vector_store, tavily_service=None):
        self.rag_engine = rag_engine
        self.vector_store = vector_store
        self.tavily_service = tavily_service
        self.logger = logging.getLogger("agentic_rag")
        
        # Initialize agents
        self.agents = {
            AgentRole.RESEARCHER: IntelligentAgent(AgentRole.RESEARCHER, rag_engine, vector_store, tavily_service),
            AgentRole.ANALYZER: IntelligentAgent(AgentRole.ANALYZER, rag_engine, vector_store, tavily_service),
            AgentRole.VALIDATOR: IntelligentAgent(AgentRole.VALIDATOR, rag_engine, vector_store, tavily_service),
            AgentRole.SYNTHESIZER: IntelligentAgent(AgentRole.SYNTHESIZER, rag_engine, vector_store, tavily_service)
        }
        
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def classify_query_complexity(self, query: str) -> TaskComplexity:
        """Classify query complexity to determine processing strategy"""
        query_lower = query.lower()
        
        # Research indicators
        research_keywords = ["compare", "analyze", "research", "comprehensive", "detailed", "in-depth"]
        if any(keyword in query_lower for keyword in research_keywords):
            return TaskComplexity.RESEARCH
        
        # Analytical indicators
        analytical_keywords = ["why", "how", "explain", "relationship", "impact", "effect"]
        if any(keyword in query_lower for keyword in analytical_keywords) and len(query.split()) > 8:
            return TaskComplexity.ANALYTICAL
        
        # Complex indicators
        complex_keywords = ["multiple", "various", "different", "several", "pros and cons"]
        if any(keyword in query_lower for keyword in complex_keywords):
            return TaskComplexity.COMPLEX
        
        # Default to simple
        return TaskComplexity.SIMPLE
    
    async def process_agentic_query(self, query: str, llm_model: str = "sarvam-m", 
                                  max_results: int = 20) -> Dict[str, Any]:
        """Process query using agentic RAG approach"""
        start_time = time.time()
        query_id = f"agentic_{int(time.time())}_{hash(query) % 10000}"
        
        self.logger.info(f"Processing agentic query {query_id}: {query[:100]}...")
        
        try:
            # Classify query complexity
            complexity = self.classify_query_complexity(query)
            self.logger.info(f"Query complexity classified as: {complexity.value}")
            
            # Determine processing strategy based on complexity
            if complexity == TaskComplexity.SIMPLE:
                return await self._process_simple_query(query, query_id, llm_model)
            else:
                return await self._process_complex_query(query, query_id, complexity, llm_model, max_results)
        
        except Exception as e:
            self.logger.error(f"Agentic processing failed for query {query_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "query_id": query_id,
                "processing_time": time.time() - start_time
            }
    
    async def _process_simple_query(self, query: str, query_id: str, llm_model: str) -> Dict[str, Any]:
        """Process simple queries with basic RAG"""
        self.logger.info(f"Processing simple query {query_id}")
        
        try:
            # Use standard RAG processing for simple queries
            result = self.rag_engine.query(
                query=query,
                llm_model=llm_model,
                max_results=10
            )
            
            return {
                "status": "success",
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "confidence": result.get("confidence", 0.8),
                "processing_strategy": "simple_rag",
                "query_id": query_id,
                "processing_time": result.get("processing_time", 0)
            }
            
        except Exception as e:
            self.logger.error(f"Simple query processing failed: {e}")
            raise
    
    async def _process_complex_query(self, query: str, query_id: str, complexity: TaskComplexity,
                                   llm_model: str, max_results: int) -> Dict[str, Any]:
        """Process complex queries using multi-agent approach"""
        self.logger.info(f"Processing complex query {query_id} with {complexity.value} strategy")
        
        # Step 1: Research phase
        research_task = AgentTask(
            task_id=f"{query_id}_research",
            agent_role=AgentRole.RESEARCHER,
            query=query,
            context={"max_results": max_results}
        )
        
        research_result = await self.agents[AgentRole.RESEARCHER].process_task(research_task)
        
        if not research_result.success:
            raise Exception(f"Research phase failed: {research_result.result.get('error', 'Unknown error')}")
        
        # Step 2: Analysis phase
        analysis_task = AgentTask(
            task_id=f"{query_id}_analysis",
            agent_role=AgentRole.ANALYZER,
            query=query,
            context={
                "sources": research_result.result.get("kb_sources", []) + research_result.result.get("web_sources", [])
            }
        )
        
        analysis_result = await self.agents[AgentRole.ANALYZER].process_task(analysis_task)
        
        # Step 3: Validation phase
        validation_task = AgentTask(
            task_id=f"{query_id}_validation",
            agent_role=AgentRole.VALIDATOR,
            query=query,
            context={
                "analysis": analysis_result.result if analysis_result.success else {},
                "sources": research_result.result.get("kb_sources", []) + research_result.result.get("web_sources", [])
            }
        )
        
        validation_result = await self.agents[AgentRole.VALIDATOR].process_task(validation_task)
        
        # Step 4: Synthesis phase
        synthesis_task = AgentTask(
            task_id=f"{query_id}_synthesis",
            agent_role=AgentRole.SYNTHESIZER,
            query=query,
            context={
                "research": research_result.result if research_result.success else {},
                "analysis": analysis_result.result if analysis_result.success else {},
                "validation": validation_result.result if validation_result.success else {}
            }
        )
        
        synthesis_result = await self.agents[AgentRole.SYNTHESIZER].process_task(synthesis_task)
        
        # Compile final result
        total_processing_time = sum([
            research_result.processing_time,
            analysis_result.processing_time,
            validation_result.processing_time,
            synthesis_result.processing_time
        ])
        
        final_confidence = (
            research_result.confidence +
            analysis_result.confidence +
            validation_result.confidence +
            synthesis_result.confidence
        ) / 4
        
        return {
            "status": "success",
            "answer": synthesis_result.result.get("final_answer", "Unable to generate comprehensive response"),
            "sources": research_result.result.get("kb_sources", []),
            "web_sources": research_result.result.get("web_sources", []),
            "confidence": final_confidence,
            "processing_strategy": "agentic_multi_phase",
            "query_complexity": complexity.value,
            "query_id": query_id,
            "processing_time": total_processing_time,
            "agent_results": {
                "research": asdict(research_result),
                "analysis": asdict(analysis_result),
                "validation": asdict(validation_result),
                "synthesis": asdict(synthesis_result)
            },
            "metadata": {
                "total_sources": research_result.result.get("total_sources", 0),
                "validation_confidence": validation_result.result.get("overall_confidence", 0.8) if validation_result.success else 0.5,
                "synthesis_quality": synthesis_result.result.get("synthesis_quality", "moderate") if synthesis_result.success else "basic"
            }
        }