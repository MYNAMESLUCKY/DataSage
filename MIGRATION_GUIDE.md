# Migration Guide: Streamlit to FastAPI + React

## Overview
This guide outlines the migration from Streamlit to a scalable FastAPI + React architecture with business logic protection, subscription system, and free GPU model integration.

## Architecture Changes

### Before (Streamlit)
```
┌─────────────────┐
│   Streamlit     │
│   Frontend      │
│   (Single Page) │
└─────────────────┘
         │
┌─────────────────┐
│   Python        │
│   Backend       │
│   (Monolithic)  │
└─────────────────┘
```

### After (FastAPI + React)
```
┌─────────────────┐    ┌─────────────────┐
│   React         │    │   FastAPI       │
│   Frontend      │◄──►│   Backend       │
│   (Scalable)    │    │   (Microservice)│
└─────────────────┘    └─────────────────┘
         │                       │
┌─────────────────┐    ┌─────────────────┐
│   Next.js       │    │   Business      │
│   Framework     │    │   Logic Layer   │
│   (Production)  │    │   (Protection)  │
└─────────────────┘    └─────────────────┘
                                │
                    ┌─────────────────┐
                    │   Subscription  │
                    │   System        │
                    │   (Billing)     │
                    └─────────────────┘
```

## Key Improvements

### 1. Business Logic Protection
- **Rate limiting**: Prevents resource abuse
- **Subscription tiers**: Free, Pro, Enterprise
- **Usage tracking**: Monitor and bill usage
- **Content filtering**: Prevent harmful queries

### 2. Free GPU Model Integration
- **Hugging Face Inference API**: Free tier models
- **Multiple providers**: Llama, Mistral, Gemma, Qwen
- **Intelligent routing**: Auto-select best model
- **Cost optimization**: Prioritize free models

### 3. Enhanced Search Capabilities  
- **Serper Dev API**: Fast, cheap Google search
- **Dual search**: Tavily + Serper for reliability
- **Rate limiting**: Prevent search abuse
- **Cost control**: Track search usage

### 4. Scalable Frontend
- **React components**: Modular, reusable
- **TypeScript**: Type safety
- **Real-time updates**: WebSocket support
- **Mobile responsive**: Tailwind CSS

## Migration Steps

### Phase 1: Backend API Development
1. **Create FastAPI endpoints** (✅ Completed)
   - `/api/v1/query/process` - Main query processing
   - `/api/v1/models/available` - List available models
   - `/api/v1/subscription/*` - Subscription management
   - `/api/v1/system/status` - System health

2. **Implement business logic** (✅ Completed)
   - Subscription system with SQLite
   - Usage tracking and billing
   - Rate limiting per tier
   - Abuse prevention system

3. **Integrate free LLM models** (✅ Completed)
   - Hugging Face API integration
   - Model selection algorithm
   - Cost tracking and optimization
   - Fallback mechanisms

### Phase 2: Frontend Development
1. **Next.js setup** (✅ Completed)
   - Project structure with TypeScript
   - Tailwind CSS for styling
   - API client configuration
   - Environment setup

2. **React components** (✅ Completed)
   - Dashboard with metrics
   - Query interface with real-time feedback
   - Subscription management
   - Usage monitoring

3. **Hooks and utilities** (✅ Completed)
   - `useRAGSystem` for query processing
   - `useSubscription` for billing
   - `apiClient` for HTTP requests
   - Error handling and retries

### Phase 3: Integration & Testing
1. **API integration** (In Progress)
   - Connect React frontend to FastAPI backend
   - Test all subscription tiers
   - Validate rate limiting
   - Ensure error handling

2. **Performance optimization**
   - Implement caching strategies
   - Optimize bundle size
   - Add performance monitoring
   - Load testing

### Phase 4: Deployment & Monitoring
1. **Production deployment**
   - Frontend: Vercel/Netlify
   - Backend: Railway/Render
   - Database: PostgreSQL
   - Monitoring: Analytics dashboard

## File Structure

### Backend (FastAPI)
```
src/backend/
├── enterprise_api.py          # Main API endpoints
├── free_llm_models.py         # Free LLM integration
├── serper_search.py           # Serper API integration  
├── subscription_system.py     # Billing and plans
├── gpu_accelerator.py         # GPU infrastructure
├── hybrid_gpu_processor.py    # GPU-accelerated processing
└── ...existing files...
```

### Frontend (React)
```
frontend/
├── src/
│   ├── components/
│   │   ├── Dashboard.tsx      # Main dashboard
│   │   ├── QueryInterface.tsx # Query processing UI
│   │   ├── SubscriptionCard.tsx
│   │   └── ...
│   ├── hooks/
│   │   ├── useRAGSystem.ts    # RAG system integration
│   │   ├── useSubscription.ts # Subscription management
│   │   └── useAuth.ts
│   ├── utils/
│   │   ├── apiClient.ts       # HTTP client
│   │   └── ...
│   └── pages/
├── package.json               # Dependencies
└── next.config.js            # Next.js configuration
```

## Business Logic Implementation

### Subscription Tiers

#### Free Tier
- **Cost**: $0/month
- **Limits**: 50 queries/day, 5 sources max
- **Features**: Basic RAG, free models only
- **Rate limit**: 10 requests/minute

#### Pro Tier  
- **Cost**: $29.99/month
- **Limits**: 2,000 queries/day, 15 sources max
- **Features**: GPU acceleration, premium models
- **Rate limit**: 60 requests/minute

#### Enterprise Tier
- **Cost**: $99.99/month  
- **Limits**: Unlimited queries and sources
- **Features**: All models, priority support, SLA
- **Rate limit**: 300 requests/minute

### Cost Optimization Strategy

#### Free Models Priority Order
1. **Hugging Face Free Tier** (Llama 3.2, Mistral 7B)
2. **Together AI Free Credits** (Llama 3.1 8B)  
3. **OpenRouter Free Models** (Various providers)
4. **Local processing** (Final fallback)

#### Search Cost Control
- **Serper Dev**: $1 per 1,000 searches
- **Free tier limit**: 20 searches/day
- **Pro tier limit**: 500 searches/day
- **Enterprise**: Unlimited

## Security Features

### Abuse Prevention
- **Content filtering**: Block harmful/illegal queries
- **Rate limiting**: Progressive penalties for violations
- **Account flagging**: Temporary blocks for repeat offenders
- **Usage monitoring**: Track patterns and anomalies

### Data Protection
- **API key security**: Server-side storage only
- **Request validation**: Input sanitization
- **Error handling**: No sensitive data in responses
- **Audit logging**: Track all API calls

## Performance Targets

### Response Times
- **Simple queries**: <1 second (with GPU)
- **Complex queries**: <3 seconds (with GPU)
- **Free tier**: <5 seconds (CPU only)
- **Enterprise**: <0.5 seconds (dedicated GPU)

### Scalability
- **Concurrent users**: 1,000+ on Pro/Enterprise
- **Query throughput**: 100+ queries/second
- **Uptime**: 99.9% SLA for Enterprise
- **Global CDN**: Sub-100ms static content

## Deployment Configuration

### Environment Variables
```bash
# Backend
HUGGINGFACE_API_KEY=hf_xxxxx
SERPER_API_KEY=xxxxx  
TOGETHER_API_KEY=xxxxx
OPENROUTER_API=xxxxx
STRIPE_SECRET_KEY=sk_xxxxx
DATABASE_URL=postgresql://xxxxx

# Frontend  
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=pk_xxxxx
```

### Docker Configuration
```dockerfile
# Backend Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "src.backend.enterprise_api:app", "--host", "0.0.0.0", "--port", "8000"]

# Frontend Dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
CMD ["npm", "start"]
```

## Migration Benefits

### Technical Benefits
- **Scalability**: Handle 10x more users
- **Performance**: 3-5x faster with GPU acceleration
- **Reliability**: Microservice architecture
- **Maintainability**: Modular codebase

### Business Benefits
- **Revenue generation**: Subscription model
- **Cost control**: Prevent resource abuse
- **User segmentation**: Tiered feature access
- **Growth potential**: Enterprise-ready platform

### Developer Benefits
- **Modern stack**: React + FastAPI
- **Type safety**: TypeScript integration
- **Developer experience**: Hot reload, debugging
- **Testing**: Component and API testing

## Next Steps

1. **Complete API integration** testing
2. **Deploy staging environment** for validation
3. **User acceptance testing** with beta users
4. **Performance optimization** and monitoring
5. **Production deployment** with CI/CD pipeline

This migration transforms the RAG system from a simple Streamlit app into a scalable, enterprise-ready platform with proper business logic, billing, and resource protection.