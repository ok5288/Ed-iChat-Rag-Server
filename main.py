# main.py
import os
import json
import tempfile
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain 导入（精简）
from langchain.vectorstores import Milvus
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# 初始化 FastAPI 应用
app = FastAPI(title="Ed-iChat RAG Server", version="1.0.0")

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置常量
MILVUS_HOST = os.getenv("MILVUS_HOST", "")
MILVUS_PORT = os.getenv("MILVUS_PORT", "443")
MILVUS_USER = os.getenv("MILVUS_USER", "root")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_RAG_TOP_K", "10"))
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_RAG_CHUNK_SIZE", "512"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_RAG_CHUNK_OVERLAP", "80"))

# 全局变量存储向量存储实例
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

def get_vectorstore(namespace: str = "default"):
    """获取或创建向量存储实例"""
    global _vectorstore
    
    if _vectorstore is None:
        try:
            # 初始化 Gemini Embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GEMINI_API_KEY
            )
            
            # 连接 Milvus
            _vectorstore = Milvus(
                embedding_function=embeddings,
                connection_args={
                    "host": MILVUS_HOST,
                    "port": MILVUS_PORT,
                    "user": MILVUS_USER,
                    "password": MILVUS_PASSWORD
                },
                collection_name=f"ed_ichat_{namespace}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to connect to Milvus: {str(e)}"
            )
    
    return _vectorstore

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    milvus_connected = False
    gemini_configured = bool(GEMINI_API_KEY and MILVUS_HOST)
    
    try:
        if gemini_configured:
            vs = get_vectorstore()
            milvus_connected = True
    except:
        pass
    
    return HealthResponse(
        status="healthy" if milvus_connected else "partial",
        milvus_connected=milvus_connected,
        gemini_configured=gemini_configured
    )

@app.post("/embed", response_model=QueryResponse)
async def embed_documents(request: QueryRequest):
    """查询文档"""
    try:
        vs = get_vectorstore(request.namespace)
        
        # 执行相似度搜索
        results = vs.similarity_search_with_score(
            request.query,
            k=request.top_k
        )
        
        # 格式化结果
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            })
        
        return QueryResponse(
            results=formatted_results,
            query=request.query,
            top_k=request.top_k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    namespace: str = "default"
):
    """上传并处理文档（仅支持纯文本）"""
    try:
        # 读取文件内容
        content = await file.read()
        text_content = content.decode("utf-8")
        
        # 创建文档对象
        doc = Document(
            page_content=text_content,
            metadata={"source": file.filename}
        )
        
        # 分割文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )
        split_documents = text_splitter.split_documents([doc])
        
        # 添加到向量存储
        vs = get_vectorstore(namespace)
        vs.add_documents(split_documents)
        
        return {
            "status": "success",
            "message": f"Uploaded {len(split_documents)} chunks from {file.filename}",
            "chunks": len(split_documents),
            "namespace": namespace
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear")
async def clear_namespace(namespace: str = "default"):
    """清空命名空间"""
    try:
        vs = get_vectorstore(namespace)
        return {
            "status": "success",
            "message": f"Namespace {namespace} cleared",
            "namespace": namespace
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats(namespace: str = "default"):
    """获取统计信息"""
    try:
        vs = get_vectorstore(namespace)
        stats = vs.col.num_entities
        return {
            "status": "success",
            "namespace": namespace,
            "total_documents": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
