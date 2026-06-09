# Semantic Cache Proxy for LLMs

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![Groq](https://img.shields.io/badge/Groq-LLM-orange.svg)](https://groq.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)

## Overview

A production‑ready **semantic caching proxy** that sits between your application and an LLM (Groq). It caches responses at two levels:
- **Exact cache (Redis)** – identical queries return instantly.
- **Semantic cache (Chroma)** – similar queries (different wording) return cached responses.

**Result:** Up to 90% reduction in token costs and latency.

## Features

 • **Two‑tier caching** – Redis for exact matches, Chroma for semantic similarity.
 • **LRU eviction** – Automatically bounds semantic cache size (no memory explosion).
 • **Token control** – Three strategies: `reject`, `truncate_with_warning`, `summarize_overflow`.
 • **Async FastAPI** – Handles high concurrency. 
 • **Retries & timeouts** – Robust error handling with exponential backoff.
 • **Docker Compose** – One‑command deployment.
 • **Structured logging** – Production‑grade observability.
 • **Health checks** – For orchestration (Kubernetes, Docker).

## Architecture

- **Exact cache:** SHA256 hash of (context + query) → response. TTL = 1 hour.
- **Semantic cache:** Embedding (all-MiniLM-L6-v2) → cosine similarity ≥ 0.92 → response. LRU eviction (max 10,000 entries).
- **Token control:** If prompt exceeds `MAX_PROMPT_TOKENS`, strategy decides:
  - `reject` → HTTP 400.
  - `truncate_with_warning` → sliding‑window truncation.
  - `summarize_overflow` → keep first 70% of limit, summarize overflow with cheap LLM.

## Prerequisites

- Docker and Docker Compose (recommended) OR Python 3.11+
- Groq API key ([free tier](https://console.groq.com))
- (Optional) Redis for local development

## Quick Start with Docker

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/semantic-cache-proxy.git
   cd semantic-cache-proxy
   
2. **Configure environment**
   cp .env.example .env
    # Edit .env and add your GROQ_API_KEY

3. **Run with Docker Compose**
   docker-compose up -d

4. **Test the API**
   curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is your return policy for defective items?"}'

## API Endpoints

POST /chat
Request body:
{
  "query": "How do I reset my router?",
  "context": "Previous conversation: user has a router model XR-100"
}

Response (cache hit):
{
  "answer": "Hold the reset button for 10 seconds...",
  "cache_hit": true,
  "cache_type": "exact",
  "tokens_sent": 0,
  "tokens_received": 0,
  "tokens_saved": 0,
  "latency_ms": 12.34,
  "truncated": false,
  "truncated_warning": null
}

Response (cache miss with truncation warning):
{
  "answer": "...",
  "cache_hit": false,
  "cache_type": null,
  "tokens_sent": 3500,
  "tokens_received": 1024,
  "tokens_saved": 0,
  "latency_ms": 2345.67,
  "truncated": true,
  "truncated_warning": "Prompt was automatically truncated: the middle part was removed to fit token limit."
}

GET /cache/stats
Returns cache sizes and configuration.

GET /health
Liveness probe for orchestration.

TRUNCATION STRATEGIES
• Strategy - reject (default)	
  Behavior - Returns HTTP 400 if prompt too long.	
  When to use - Legal, medical, financial – no data loss allowed.

• Strategy - truncate_with_warning	
  Behavior - Automatically truncates (keep first X%, last Y%). Returns warning.	
  When to use - General chat where losing middle is acceptable.

• Strategy - summarize_overflow
  Behavior - SummariZes overflow using cheap LLM, prepends summary. Returns warning.	
  When to use - Long documents, preserving all information.