#!/usr/bin/env python3
"""
Coding Ground API Backend
Separate API server for AI coding assistance
Uses DeepSeek R1 and Qwen3 Coder models with internet documentation access
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import asyncio
import aiohttp
import json
import time
import os
import logging
import jwt
from datetime import datetime, timedelta
import requests
import trafilatura

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Coding Ground API",
    description="AI-Powered Coding Assistant with Documentation Access",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "coding-ground-secret-key-2025")

# API Models
class CodeGenerationRequest(BaseModel):
    prompt: str
    model: str
    language: str = "python"
    context: str = ""
    include_docs: bool = True
    search_resources: bool = True

class CodeExplanationRequest(BaseModel):
    code: str
    model: str
    include_docs: bool = True

class CodeFixRequest(BaseModel):
    code: str
    error: str
    model: str
    search_docs: bool = True

class APIResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str = datetime.now().isoformat()

# AI Model Configuration
MODELS_CONFIG = {
    "deepseek-r1": {
        "base_url": "https://api.openrouter.ai/api/v1",
        "api_key": os.getenv("deepseek_r1_api"),
        "model_name": "deepseek/deepseek-r1",
        "description": "Advanced reasoning for complex coding problems"
    },
    "qwen3-coder-7b": {
        "base_url": "https://api.openrouter.ai/api/v1",
        "api_key": os.getenv("qwen_api"),
        "model_name": "qwen/qwen-2.5-coder-7b-instruct:free",
        "description": "Efficient coding assistant"
    },
    "qwen3-coder-14b": {
        "base_url": "https://api.openrouter.ai/api/v1", 
        "api_key": os.getenv("qwen_api"),
        "model_name": "qwen/qwen-2.5-coder-14b-instruct:free",
        "description": "Advanced coding assistant"
    }
}

class CodingAIEngine:
    """AI engine for coding assistance with documentation access"""
    
    def __init__(self):
        self.web_searcher = WebDocumentationSearcher()
        
    async def generate_code(self, request: CodeGenerationRequest) -> Dict[str, Any]:
        """Generate code with documentation support"""
        try:
            # Search for relevant documentation if requested
            docs_context = ""
            if request.include_docs or request.search_resources:
                docs_context = await self.web_searcher.search_programming_docs(
                    query=request.prompt,
                    language=request.language
                )
            
            # Prepare enhanced prompt
            enhanced_prompt = self._build_code_generation_prompt(
                request.prompt,
                request.language,
                request.context,
                docs_context
            )
            
            # Generate code using selected model
            response = await self._call_ai_model(request.model, enhanced_prompt)
            
            return {
                "code": self._extract_code_from_response(response),
                "explanation": self._extract_explanation_from_response(response),
                "documentation_used": bool(docs_context),
                "model_used": request.model
            }
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")
    
    async def explain_code(self, request: CodeExplanationRequest) -> Dict[str, Any]:
        """Explain existing code with documentation context"""
        try:
            # Search for relevant documentation
            docs_context = ""
            if request.include_docs:
                # Extract key concepts from code for documentation search
                concepts = self._extract_coding_concepts(request.code)
                docs_context = await self.web_searcher.search_programming_docs(
                    query=" ".join(concepts),
                    language="general"
                )
            
            # Prepare explanation prompt
            prompt = self._build_explanation_prompt(request.code, docs_context)
            
            # Get explanation from AI model
            response = await self._call_ai_model(request.model, prompt)
            
            return {
                "explanation": response,
                "documentation_used": bool(docs_context),
                "model_used": request.model
            }
            
        except Exception as e:
            logger.error(f"Code explanation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Code explanation failed: {str(e)}")
    
    async def fix_code(self, request: CodeFixRequest) -> Dict[str, Any]:
        """Fix code based on error with documentation support"""
        try:
            # Search for error-specific documentation
            docs_context = ""
            if request.search_docs:
                docs_context = await self.web_searcher.search_error_solutions(
                    error=request.error,
                    code=request.code
                )
            
            # Prepare fix prompt
            prompt = self._build_fix_prompt(request.code, request.error, docs_context)
            
            # Get fixed code from AI model
            response = await self._call_ai_model(request.model, prompt)
            
            return {
                "fixed_code": self._extract_code_from_response(response),
                "explanation": self._extract_explanation_from_response(response),
                "documentation_used": bool(docs_context),
                "model_used": request.model
            }
            
        except Exception as e:
            logger.error(f"Code fixing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Code fixing failed: {str(e)}")
    
    async def _call_ai_model(self, model_id: str, prompt: str) -> str:
        """Call the specified AI model"""
        if model_id not in MODELS_CONFIG:
            raise ValueError(f"Unknown model: {model_id}")
        
        config = MODELS_CONFIG[model_id]
        
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": config["model_name"],
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert coding assistant with access to comprehensive programming documentation. Provide accurate, well-explained code solutions."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.3
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{config['base_url']}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    raise Exception(f"API call failed: {response.status} - {error_text}")
    
    def _build_code_generation_prompt(self, prompt: str, language: str, context: str, docs: str) -> str:
        """Build enhanced prompt for code generation"""
        return f"""
Generate {language} code for the following request:

REQUEST: {prompt}

EXISTING CONTEXT:
{context if context else "No existing context"}

RELEVANT DOCUMENTATION:
{docs if docs else "No specific documentation found"}

Please provide:
1. Clean, working {language} code
2. Clear explanation of the implementation
3. Best practices and considerations
4. Usage examples if applicable

Format your response with:
```{language}
[YOUR CODE HERE]
```

EXPLANATION:
[YOUR EXPLANATION HERE]
"""
    
    def _build_explanation_prompt(self, code: str, docs: str) -> str:
        """Build prompt for code explanation"""
        return f"""
Explain the following code in detail:

CODE:
```
{code}
```

RELEVANT DOCUMENTATION:
{docs if docs else "No specific documentation found"}

Please provide:
1. Overall purpose and functionality
2. Step-by-step breakdown
3. Key concepts and patterns used
4. Potential improvements or considerations
"""
    
    def _build_fix_prompt(self, code: str, error: str, docs: str) -> str:
        """Build prompt for code fixing"""
        return f"""
Fix the following code that has an error:

ORIGINAL CODE:
```
{code}
```

ERROR:
{error}

RELEVANT DOCUMENTATION:
{docs if docs else "No specific documentation found"}

Please provide:
1. The fixed code
2. Explanation of what was wrong
3. Why your fix resolves the issue

Format your response with:
```
[FIXED CODE HERE]
```

EXPLANATION:
[YOUR EXPLANATION HERE]
"""
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract code blocks from AI response"""
        lines = response.split('\n')
        in_code_block = False
        code_lines = []
        
        for line in lines:
            if line.strip().startswith('```'):
                if in_code_block:
                    break
                else:
                    in_code_block = True
                    continue
            
            if in_code_block:
                code_lines.append(line)
        
        return '\n'.join(code_lines) if code_lines else response
    
    def _extract_explanation_from_response(self, response: str) -> str:
        """Extract explanation from AI response"""
        if "EXPLANATION:" in response:
            return response.split("EXPLANATION:")[-1].strip()
        elif "```" in response:
            # If there's code, explanation is usually after
            parts = response.split("```")
            if len(parts) > 2:
                return parts[-1].strip()
        
        return response
    
    def _extract_coding_concepts(self, code: str) -> List[str]:
        """Extract key programming concepts from code"""
        # Simple keyword extraction for documentation search
        keywords = []
        
        # Common programming concepts
        if "import " in code:
            keywords.append("import")
        if "class " in code:
            keywords.append("class")
        if "def " in code:
            keywords.append("function")
        if "async " in code:
            keywords.append("async")
        if "await " in code:
            keywords.append("await")
        
        return keywords[:5]  # Limit to top 5 concepts


class WebDocumentationSearcher:
    """Search web for programming documentation and resources"""
    
    async def search_programming_docs(self, query: str, language: str) -> str:
        """Search for programming documentation"""
        try:
            # Enhanced search query for programming docs
            search_query = f"{query} {language} programming documentation examples tutorial"
            
            # Search authoritative sources
            results = await self._search_web(search_query, max_results=3)
            
            if results:
                return self._format_documentation(results)
            
        except Exception as e:
            logger.error(f"Documentation search failed: {e}")
        
        return ""
    
    async def search_error_solutions(self, error: str, code: str) -> str:
        """Search for solutions to specific errors"""
        try:
            # Extract error type for better search
            error_type = error.split('\n')[0] if '\n' in error else error
            search_query = f"{error_type} solution fix programming"
            
            results = await self._search_web(search_query, max_results=2)
            
            if results:
                return self._format_documentation(results)
                
        except Exception as e:
            logger.error(f"Error solution search failed: {e}")
        
        return ""
    
    async def _search_web(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Perform web search for documentation"""
        try:
            # Use Tavily for web search if available
            if os.getenv("TAVILY_API_KEY"):
                return await self._search_with_tavily(query, max_results)
            else:
                # Fallback to direct documentation sources
                return await self._search_known_sources(query)
                
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []
    
    async def _search_with_tavily(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using Tavily API"""
        try:
            from tavily import TavilyClient
            
            client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            results = client.search(
                query=query,
                max_results=max_results,
                include_domains=["docs.python.org", "developer.mozilla.org", "stackoverflow.com", "github.com"]
            )
            
            formatted_results = []
            for result in results.get("results", []):
                formatted_results.append({
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "url": result.get("url", "")
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return []
    
    async def _search_known_sources(self, query: str) -> List[Dict[str, Any]]:
        """Search known documentation sources"""
        # Fallback to known documentation URLs
        known_sources = [
            "https://docs.python.org/3/",
            "https://developer.mozilla.org/en-US/docs/",
            "https://docs.github.com/"
        ]
        
        results = []
        for source in known_sources[:2]:  # Limit to 2 sources
            try:
                content = trafilatura.fetch_url(source)
                if content:
                    text = trafilatura.extract(content)
                    if text and query.lower() in text.lower():
                        results.append({
                            "title": f"Documentation from {source}",
                            "content": text[:1000],  # Limit content
                            "url": source
                        })
            except:
                continue
        
        return results
    
    def _format_documentation(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for AI context"""
        formatted = []
        
        for i, result in enumerate(results[:3], 1):
            formatted.append(f"""
SOURCE {i}: {result.get('title', 'Documentation')}
URL: {result.get('url', 'N/A')}
CONTENT: {result.get('content', '')[:500]}...
""")
        
        return "\n".join(formatted)


# Initialize the coding AI engine
coding_engine = CodingAIEngine()

# JWT token functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "coding-ground-api", "timestamp": datetime.now().isoformat()}

@app.post("/auth/token")
async def create_token(user_id: str, role: str = "developer"):
    """Create authentication token"""
    access_token_expires = timedelta(hours=24)
    access_token = create_access_token(
        data={"sub": user_id, "role": role}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/code/generate")
async def generate_code_endpoint(
    request: CodeGenerationRequest,
    current_user: dict = Depends(verify_token)
):
    """Generate code using AI models with documentation access"""
    try:
        result = await coding_engine.generate_code(request)
        return APIResponse(success=True, data=result)
    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        return APIResponse(success=False, error=str(e))

@app.post("/code/explain")
async def explain_code_endpoint(
    request: CodeExplanationRequest,
    current_user: dict = Depends(verify_token)
):
    """Explain existing code with documentation context"""
    try:
        result = await coding_engine.explain_code(request)
        return APIResponse(success=True, data=result)
    except Exception as e:
        logger.error(f"Code explanation failed: {e}")
        return APIResponse(success=False, error=str(e))

@app.post("/code/fix")
async def fix_code_endpoint(
    request: CodeFixRequest,
    current_user: dict = Depends(verify_token)
):
    """Fix code based on error with documentation support"""
    try:
        result = await coding_engine.fix_code(request)
        return APIResponse(success=True, data=result)
    except Exception as e:
        logger.error(f"Code fixing failed: {e}")
        return APIResponse(success=False, error=str(e))

@app.get("/models")
async def list_models():
    """List available AI models"""
    return {
        "models": [
            {
                "id": model_id,
                "name": config["model_name"],
                "description": config["description"]
            }
            for model_id, config in MODELS_CONFIG.items()
        ]
    }

class ProjectRequest(BaseModel):
    name: str
    description: str
    language: str = "python"
    framework: str = ""
    features: List[str] = []

class CodeReviewRequest(BaseModel):
    code: str
    language: str = "python"
    focus_areas: List[str] = ["performance", "security", "best_practices"]

@app.post("/projects/generate")
async def generate_project_endpoint(
    request: ProjectRequest,
    current_user: dict = Depends(verify_token)
):
    """Generate a complete project structure with multiple files"""
    try:
        # Enhanced prompt for project generation
        prompt = f"""
Create a complete {request.language} project for: {request.name}

DESCRIPTION: {request.description}
FRAMEWORK: {request.framework or 'Standard library'}
FEATURES: {', '.join(request.features) if request.features else 'Basic functionality'}

Generate a project structure with:
1. Main application file
2. Configuration files
3. Requirements/dependencies
4. README with setup instructions
5. Example usage
6. Basic tests

Provide the complete file structure and contents.
"""
        
        # Search for relevant documentation
        docs_context = await coding_engine.web_searcher.search_programming_docs(
            query=f"{request.language} {request.framework} project structure best practices",
            language=request.language
        )
        
        enhanced_prompt = coding_engine._build_code_generation_prompt(
            prompt, request.language, "", docs_context
        )
        
        # Use the most advanced model for project generation
        response = await coding_engine._call_ai_model("deepseek-r1", enhanced_prompt)
        
        return APIResponse(success=True, data={
            "project_structure": response,
            "language": request.language,
            "framework": request.framework,
            "documentation_used": bool(docs_context)
        })
        
    except Exception as e:
        logger.error(f"Project generation failed: {e}")
        return APIResponse(success=False, error=str(e))

@app.post("/code/review")
async def code_review_endpoint(
    request: CodeReviewRequest,
    current_user: dict = Depends(verify_token)
):
    """Perform comprehensive code review with best practices analysis"""
    try:
        # Search for relevant code review guidelines
        docs_context = await coding_engine.web_searcher.search_programming_docs(
            query=f"{request.language} code review best practices security performance",
            language=request.language
        )
        
        # Build comprehensive review prompt
        prompt = f"""
Perform a comprehensive code review for the following {request.language} code:

CODE TO REVIEW:
```{request.language}
{request.code}
```

FOCUS AREAS: {', '.join(request.focus_areas)}

RELEVANT DOCUMENTATION:
{docs_context if docs_context else "Standard best practices"}

Please provide:
1. SECURITY ANALYSIS - Identify potential vulnerabilities
2. PERFORMANCE REVIEW - Suggest optimizations
3. CODE QUALITY - Check style, readability, maintainability
4. BEST PRACTICES - Verify adherence to language conventions
5. SUGGESTIONS - Specific improvements with examples
6. OVERALL SCORE - Rate the code quality (1-10)

Format your response with clear sections and actionable recommendations.
"""
        
        response = await coding_engine._call_ai_model("deepseek-r1", prompt)
        
        return APIResponse(success=True, data={
            "review": response,
            "language": request.language,
            "focus_areas": request.focus_areas,
            "documentation_used": bool(docs_context)
        })
        
    except Exception as e:
        logger.error(f"Code review failed: {e}")
        return APIResponse(success=False, error=str(e))

@app.get("/documentation/search")
async def search_documentation_endpoint(
    query: str,
    language: str = "python",
    current_user: dict = Depends(verify_token)
):
    """Search programming documentation and resources"""
    try:
        results = await coding_engine.web_searcher.search_programming_docs(
            query=query,
            language=language
        )
        
        return APIResponse(success=True, data={
            "query": query,
            "language": language,
            "documentation": results,
            "sources_searched": ["Official docs", "Stack Overflow", "GitHub", "MDN"]
        })
        
    except Exception as e:
        logger.error(f"Documentation search failed: {e}")
        return APIResponse(success=False, error=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)