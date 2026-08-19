# main.py
import os
import logging
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from langchain.vectorstores import Milvus
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document
    IMPORTS_OK = True
except ImportError as e:
    logger.error(f"Import error: {e}")
    IMPORTS_OK = False

app = FastAPI(title="Ed-iChat RAG Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 环境变量配置
MILVUS_HOST = os.getenv("MILVUS_HOST", "")
MILVUS_PORT = os.getenv("MILVUS_PORT", "443")
MILVUS_USER = os.getenv("MILVUS_USER", "root")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_RAG_TOP_K", "10"))
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_RAG_CHUNK_SIZE", "512"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_RAG_CHUNK_OVERLAP", "80"))

_vectorstore = None

class QueryRequest(BaseModel):
    query: str
    top_k: int = DEFAULT_TOP_K
    namespace: str = "default"

class QueryResponse(BaseModel):
    results: List[dict]
    query: str
    top_k: int

class HealthResponse(BaseModel):
    status: str
    milvus_connected: bool
    gemini_configured: bool

@app.get("/")
async def root():
    logger.info("Root endpoint called!")
    return {
        "message": "Ed-iChat RAG Server is running!",
        "endpoints": {
            "health": "/health",
            "embed": "/embed",
            "upload": "/upload",
            "stats": "/stats",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        milvus_connected = False
        gemini_configured = bool(GEMINI_API_KEY and MILVUS_HOST)
        
        if gemini_configured:
            vs = get_vectorstore()
            milvus_connected = True
        
        return HealthResponse(
            status="healthy" if milvus_connected else "partial",
            milvus_connected=milvus_connected,
            gemini_configured=gemini_configured
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="error",
            milvus_connected=False,
            gemini_configured=bool(GEMINI_API_KEY and MILVUS_HOST)
        )

@app.post("/embed", response_model=QueryResponse)
async def embed_documents(request: QueryRequest):
    try:
        vs = get_vectorstore(request.namespace)
        results = vs.similarity_search_with_score(request.query, k=request.top_k)
        
        formatted_results = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            }
            for doc, score in results
        ]
        
        return QueryResponse(results=formatted_results, query=request.query, top_k=request.top_k)
    except Exception as e:
        logger.error(f"Embed error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), namespace: str = "default"):
    try:
        content = await file.read()
        text_content = content.decode("utf-8")
        
        doc = Document(page_content=text_content, metadata={"source": file.filename})
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )
        split_documents = text_splitter.split_documents([doc])
        
        vs = get_vectorstore(namespace)
        vs.add_documents(split_documents)
        
        return {
            "status": "success",
            "message": f"Uploaded {len(split_documents)} chunks from {file.filename}",
            "chunks": len(split_documents),
            "namespace": namespace
        }
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats(namespace: str = "default"):
    try:
        vs = get_vectorstore(namespace)
        return {
            "status": "success",
            "namespace": namespace,
            "total_documents": vs.col.num_entities
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def get_vectorstore(namespace: str = "default"):
    global _vectorstore
    
    if not IMPORTS_OK:
        raise ImportError("Failed to import required packages")
    
    if _vectorstore is None:
        try:
            logger.info("Initializing Milvus connection...")
            logger.info(f"Host: {MILVUS_HOST}")
            logger.info(f"Port: {MILVUS_PORT}")
            logger.info(f"User: {MILVUS_USER}")
            logger.info(f"Password length: {len(MILVUS_PASSWORD)}")
            
            from pymilvus import connections
            conn = connections.connect(
                alias="default",
                host=MILVUS_HOST,
                port=MILVUS_PORT,
                user=MILVUS_USER,
                password=MILVUS_PASSWORD
            )
            logger.info("Milvus connection test passed!")
            
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GEMINI_API_KEY
            )
            logger.info("Gemini embeddings initialized")
            
            _vectorstore = Milvus(
                embedding_function=embeddings,
                connection_args={
                    "host": MILVUS_HOST,
                    "port": MILVUS_PORT,
                    "user": MILVUS_USER,
                    "password": MILVUS_PASSWORD
                },
                collection_name=f"edichat_{namespace}"
            )
            logger.info(f"Vectorstore created for namespace: {namespace}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Milvus connection failed: {str(e)}"
            )
    
    return _vectorstore

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
