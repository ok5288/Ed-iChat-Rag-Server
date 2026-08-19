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

```bash
# 安装 Vercel CLI
npm i -g vercel

# 登录
vercel login

# 部署
vercel

# 生产部署
vercel --prod
