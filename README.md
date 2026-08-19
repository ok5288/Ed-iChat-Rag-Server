# Ed-iChat-Rag-Server
Ed-iChat-Rag-Server

轻量级 RAG Server，用于 Ed-iChat 项目的向量存储和文档检索。

## 功能

- 向量存储（Milvus / Zilliz Cloud）
- 文档上传和处理
- 相似度搜索
- Gemini Embeddings

## API 端点

- `GET /health` - 健康检查
- `POST /embed` - 查询文档
- `POST /upload` - 上传文档
- `DELETE /clear` - 清空命名空间
- `GET /stats` - 获取统计信息

## 部署到 Vercel
### 一键部署到Vercel
<div align="left">

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/ok5288/Ed-iChat-rag-server&env=MILVUS_HOST,MILVUS_PORT,MILVUS_USER,MILVUS_PASSWORD,GEMINI_API_KEY,DEFAULT_RAG_TOP_K,DEFAULT_RAG_CHUNK_SIZE,DEFAULT_RAG_CHUNK_OVERLAP&project-name=Ed-ichat-rag-server&repository-name=neo-chat-rag-server)

</div>


```bash
# 安装 Vercel CLI
npm i -g vercel

# 登录
vercel login

# 部署
vercel

# 生产部署
vercel --prod
