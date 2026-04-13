# KrishiRakshak – Agentic AI System (Hindi + English)

## Objective

Build an Agentic AI application for farmers to:
1. Detect crop diseases via image upload
2. Query government documents (PDFs)
3. Ask questions in Hindi or English
4. Receive responses in the same language along with audio output

This project is designed as a student demo to explain production-grade Agentic AI systems.

---

## Supported Crops and Diseases

Crops:
- Tomato
- Potato
- Corn

Diseases:
- Corn Common Rust
- Tomato Leaf Mold
- Tomato Early Blight
- Tomato Late Blight
- Potato Late Blight

---

## Core Capabilities

### 1. Image Diagnosis
- Input: Crop image
- Output:
  - Disease classification
  - Confidence score
  - Suggested remedy

Model is pre-finetuned and provided.

---

### 2. Document Understanding (RAG)

User can upload:
- Government policy PDFs
- Crop disease documents

System will:
- Extract text (Hindi/English)
- Chunk documents
- Generate embeddings
- Retrieve relevant information

---

## Retrieval Strategy

### Case 1: Small Document

If document size fits within LLM context window:
- Pass full document directly to Claude Sonnet
- Do not use vector database

---

### Case 2: Large Document

If document is large:
- Apply RAG pipeline:
  - Chunk using indic_rag_chunker
  - Embed using BGE-M3
  - Store in FAISS
  - Retrieve top-k
  - Rerank using Cohere

---

## Architecture

User Input:
- Image / PDF / Text

Processing Flow:
- Agent receives input
- Agent decides which tool to use
- Tool executes
- Output passed to LLM
- Final response generated
- Audio generated

---

## Tech Stack (AWS Compatible)

- LLM: Claude Sonnet
- Embeddings: BGE-M3
- Reranker: Cohere
- Chunking: indic_rag_chunker
- Audio: Amazon TTS
- Vector Store: FAISS (initial)

---

## Agent Framework

### Definition

Agent consists of:
- LLM (Claude Sonnet)
- Set of tools
- Decision-making loop

---

### Agent Loop

User Input
→ LLM decides action
→ Tool is called
→ Tool output returned
→ LLM decides next step
→ Repeat until final answer

---

### Agent Type

ReAct-style (Reason + Act)

Steps:
1. Analyze input
2. Select tool
3. Execute tool
4. Observe output
5. Continue or finalize

---

## Tools

- image_diagnosis_tool
- document_ingestion_tool
- retriever_tool
- reranker_tool
- direct_context_tool
- web_search_tool
- audio_generation_tool

---

## Tool Definitions

### Image Diagnosis Tool
- Input: image
- Output: disease, confidence

---

### Document Ingestion Tool
Steps:
- Extract text from PDF
- Normalize text
- Detect language
- Chunk using indic_rag_chunker
- Generate embeddings (BGE-M3)
- Store in FAISS

---

### Retriever Tool
Steps:
- Encode query (BGE-M3)
- Retrieve top-k documents
- Apply reranking (Cohere)

---

### Direct Context Tool
- Used for small documents
- Sends full content to LLM
- No embeddings or retrieval

---

### Web Search Tool
Purpose:
- Fetch latest government policies
- Retrieve external agricultural knowledge

Usage:
- Trigger when:
  - No relevant documents found
  - Query requires latest information

Constraint:
- Prefer trusted sources (government / ICAR)

---

### Response Generator
- Uses Claude Sonnet
- Generates grounded answers
- Maintains user language

---

### Audio Generation Tool
- Converts response to speech
- Uses Amazon TTS

---

## Additional Tools

### Query Expansion Tool
- Improves retrieval using synonyms or language variants

### Language Detection Tool
- Detects Hindi or English input

### Metadata Filter Tool
- Filters documents by crop, disease, or language

### Validation Tool
- Ensures answer is grounded in retrieved context

### Confidence Scoring Tool
- Combines model confidence and retrieval score

---

## Agent Routing Logic

IF input = image:
    → Image Diagnosis Tool

ELIF input = PDF:
    → Document Ingestion Tool

ELIF document is small:
    → Direct Context Tool

ELIF retrieval fails:
    → Web Search Tool

ELSE:
    → Retriever + Reranker

---

## Agent Framework Options

### Phase 1 (Current Approach)

- Custom agent loop (Python)
- Explicit tool calling
- Full control and transparency
- ReAct / Reflective kind of patterns

---

### Phase 2 (Optional)

- LangChain
- LlamaIndex

---

## Language Handling

- Detect input language (Hindi or English)
- Maintain same language in response
- Avoid unnecessary translation
- Use multilingual embeddings

---

## Development Plan

1. Work in notebook (reference: model_tuning.ipynb)
2. Build modules step-by-step:
   - ingestion
   - chunking
   - embedding
   - retrieval
   - reranking
   - generation
3. Test each module independently
4. Create strong test cases
5. Integrate into a single pipeline
6. Deploy on AWS

---

## Testing Strategy

Must include:

- Hindi query → Hindi response
- English query → English response
- Cross-language retrieval
- Image classification accuracy
- PDF ingestion correctness
- RAG grounding validation
- Small vs large document handling

---

## Constraints

- Do not use translation-first pipeline
- Always use "query:" and "passage:" format for embeddings
- Ensure chunk overlap
- Always apply reranking
- Avoid unnecessary complexity

---

## Role of Claude

- Work step-by-step
- Do not skip steps
- Ask for clarification if needed
- Provide minimal and correct code
- Focus on correctness
- Help design and validate test cases

---

## Final Goal

A unified system where:

User uploads image, document, or query  
→ Agent selects appropriate tool  
→ System retrieves or processes information  
→ Response generated in Hindi or English  
→ Audio output provided