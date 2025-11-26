# 昇腾Ascend RAGSDK 基本框架部署流程

***

![image-20251126152811059](https://cdn.jsdelivr.net/gh/kiu795/pic@main/img/image-20251126152811059.png)

## 1. Reranker&Embedding

这两个服务可以使用同一个镜像中的两个不同模型。

### 1.1 Embedding

#### 1.1.1Embedding基础概念

负责将文本转换为高维向量（数值数组）。是RAG系统的核心组件之一。

```python
# 文本 -> Embedding 向量
文本: "深度学习是机器学习的一个分支"
↓
Embedding 模型
↓
向量: [0.23, -0.45, 0.67, ..., 0.12]  # 通常是 384、768 或 1536 维
```

+ 语义理解和表示
  + 语义相似的文本->向量距离近
+ 文档索引（离线阶段）
+ 查询理解（在线阶段）
+ 相似度检索
  + 通过向量相似度找到最相关的文档。
+ 多语言支持
  + 某些Embedding模型支持多语言

#### 1.1.2 Embedding服务的部署与启动

+ 下载昇腾镜像仓库中的TEI镜像

![image-20251126161048051](https://cdn.jsdelivr.net/gh/kiu795/pic@main/img/image-20251126161048051.png)

+ 下载模型权重
  + Embedding->BAAI/bge-base-zh-v1.5
  + Reranker->BAAI/bge-reranker-large

+ 启动服务

  + 宿主机中启动：

    ```bash
    docker run -u <user> -e ENABLE_BOOST=True -e ASCEND_VISIBLE_DEVICES=0 -itd --name=tei --net=host \
    -v <model_dir>:/home/HwHiAiUser/model \
    <image_id> <model_id> <listen_ip> <listen_port>
    ```

  + 容器中启动：

    cd进`/home/HwHiAiUser/`，<u>*两种服务的启动方式不同点只有端口和所使用模型的id，模型的id就是模型所在文件夹的名称*</u>。

    ```bash
    ./start.sh <image_id> <model_id> <listen_ip> <listen_port>
    ```

#### 1.1.3 测试Embedding服务

```bash
curl -X POST http://127.0.0.1:7999/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["测试"]}' \
  -w "\nHTTP status: %{http_code}\n"
```

output:

```bash
[[0.014985706,-0.025844539,-0.040933415,0.03195747,-0.013863713,0.0029000952,0.027392115,0.019202854,-0.02865597,0.007299406,0.033401873,-0.020363536,-0.0028710782,-0.022478558,-0.0010575111,0.027211566,0.056383397,0.0039302013,0.052617624,-0.048000686,0.016958866,-0.0066481335,-0.026566742,-0.06484348,0.07939071,-0.05545485,-0.023703724,-0.008427847,0.027031016,-0.013554197,-0.010691179,0.02723736,0.011103867,0.010123734,-0.034717314,0.014908327,-0.050425224,0.011722897,0.029739276,0.012174274,0.018880442,-0.024387237,0.006174188,0.014650397,0.0052585383,0.033092357,-0.065101415,-0.03335029,-0.020002436,0.045395598,-0.0060774647,0.003652927,0.0446476,0.0011026488,-0.018506443,0.04926454,0.011761587,-0.050683152,-0.008247297,-0.013760541,-0.019654231,0.0838271,-0.010478388,-0.009511151,0.011065177,-0.02492889,0.04077866,
...
-0.025135232,-0.0045556803,-0.009910942]]
HTTP status: 200
```

输出高维向量和`HTTP status`

### 1.2 Reranker

#### 1.2.1 Reranker基础概念

向量检索/BM25负责从海量文档中找几十条“可能相关”的文档，但这些方法不能完全找出真正反映语义关系的内容，这时就需要Reranker进行第二层筛选。

Reranker 通常是一个跨编码器模型，例如：

+ BERT 交叉编码器

+ Cohere Rerank

+ bge-reranker-large

它把[查询+文档]一起输入到模型中，判断语义是否真正匹配。能够真正深度的理解文本，大幅减少模型幻觉，但是召回速度稍慢。

```python
用户 Query
      ↓
向量检索 / BM25 找到 top 50 条候选
      ↓
Reranker 精排序 → top 5
      ↓
把 top 5 发送给 LLM 生成回答
```

#### 1.2.2 Reranker服务的部署与启动

+ 参考Embedding。

#### 1.2.3 测试Reranker服务

```bash
curl -X POST http://127.0.0.1:7998/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "人工智能是什么？",
    "texts": [
      "人工智能是模拟人类智能的计算机系统。",
      "香蕉是一种水果。",
      "机器学习是人工智能的一个子领域。"
    ]
  }' \
  -w "\n✅ Status: %{http_code}\n"
```

output:

```bash
[{"index":0,"score":0.9992267},{"index":2,"score":0.19961986},{"index":1,"score":0.00007602479}]
✅ Status: 200
```

Reranker会按顺序返回相关文档和评分，越相关的文档评分越高，反之越低。

***

## 2. Milvus

### 2.1 Milvus 基础概念

Milvus是一个开源、专门用来管理向量数据的数据库：

+ 把文本、图片等数据做成向量（Embedding）
+ Milvus用来存储、索引和快速检索这些向量

它是构建RAG、推荐系统、相似度搜索的常用工具

### 2.2 Milvus 核心概念

#### 2.2.1 Collection

类似传统数据库里的“表”，用来存放一类向量数据。

例：
 `clothes_embedding_collection` （存衣服图片向量）
 `document_embeddings`（存文本向量）

#### 2.2.2 Field

类似数据库的字段。
 常见字段有：

| 字段类型   | 说明                  |
| ---------- | --------------------- |
| `id`       | 主键（int 或 string） |
| `vector`   | 向量字段（必须）      |
| `text`     | 原文文本（可选）      |
| `metadata` | 额外信息（可选）      |

例：

- `embedding` 是一个 768 维的向量
- `text` 是原始内容
- `doc_id` 是文档源头

#### 2.2.3 Index

为了加速向量搜索，需要在 `vector` 字段上建立索引。

Milvus 支持很多种索引：

- **IVF_FLAT**
- **IVF_SQ8**
- **HNSW**（最常用）
- **DISKANN**

索引决定了检索效率和精度：

- **HNSW：高精度，中速**
- **IVF 系列：快，但可能牺牲一点精度**
- **DISKANN：支持海量数据，适合大规模 RAG**

#### 2.2.4 Search

Milvus 支持几种常见的相似度搜索：

- **cosine 相似度**
- **L2 距离（欧式距离）**
- **内积（dot product）**

通过 Search，你输入一个 query 向量，Milvus 会找到：

```
最相似的 top-k 向量
```

用于 RAG 时，这一步就是 **找相关文档**。

#### 2.2.5 Partition

在一个 Collection 下再细分子集。

例如：

- 2023 年的文档
- 2024 年的文档
- 不同语言的文档

主要用于大规模数据下的加速搜索。

### 2.3 Milvus基本流程

```python
原始文本/图片
        ↓
Embedding 模型（如 bge-large）
        ↓
得到向量
        ↓
写入 Milvus（Insert）
        ↓
Milvus 建索引（Index）
        ↓
向量搜索（Search）
```

### 2.4 Milvus的部署

参照[milvus官网](https://milvus.io/docs/install_standalone-docker.md)(ctrl+click to visit)中的步骤，使用脚本文件创建容器，并在容器中安装Milvus。

Milvus Standalone 在主机上会开启以下端口：

| 主机端口  | 服务                    | 是否需要用               |
| --------- | ----------------------- | ------------------------ |
| **19530** | gRPC 主服务端口         | ✔ 必须使用（客户端连接） |
| **9091**  | HTTP 健康检查 / metrics | ✔ 可选                   |
| **2379**  | 内嵌 ETCD               | ✖ 一般不用               |

***

## 3. LLM服务

*此处使用`Mindie`服务调用LLM。*

### 3.1 Mindie基础概念

MindIE是一个“推理服务”，负责把模型封装成服务端接口，对外提供高速推理能力。

```python
加载模型：读取.om模型文件，将模型加载进NPU内存，创建Context,初始化底层资源
	↓
调度NPU
	↓
提供推理API
    ↓
返回结果
```

#### 3.1.1 加载模型

MindIE 会：

- 读取已经通过 ATC 转换好的 **.om 模型文件**
- 把模型加载进昇腾 NPU 的内存
- 创建模型推理上下文（Context）
- 初始化 TSD、ACL 这些底层资源

解决的问题是：
 ✔ 不用自己手动写 ACL 推理代码
 ✔ 不用关系 operator/kernel 调度
 ✔ 不用自己管理 NPU Tensor 申请/释放

#### 3.1.2 启动推理引擎

MindIE 会创建一个推理实例：

- 注册算子
- 建立推理图 session
- 分配 device memory
- 调用昇腾的 runtime 完成编译后的图加载

只要有模型，它会自动让它进入“可推理状态”。

#### 3.1.3 负责高性能的推理加速

MindIE 负责：

- 管理 NPU 上的执行流（stream）
- 调度算子
- 异步排队
- 内存复用
- 多线程推理

#### 3.1.4 提供统一API接口

MindIE 会对外暴露 REST 或 WebSocket 服务，让你通过 HTTP 调用推理：

常见 API：

```
/infer
/health
/model/list
```

使用方式示例：

```
POST http://ip:port/mindie/v1/inference
```

提交输入，返回输出。

#### 3.1.5 自动管理模型生命名周期

MindIE 会自动管理：

- 模型加载（load）
- 模型卸载（unload）
- 模型热更新
- 多模型管理
- 服务重启后的模型恢复

只需要告诉它模型在哪，它帮助一直维持服务可用。

#### 3.1.6  多模型并行

现在在 RAG 场景中可能需要：

- Embedding 模型（bge）
- Reranker 模型
- 大模型（如 LLaMA、Qwen 的 om 版）
- 分类器/判别模型

MindIE 能：

✔ 同时加载
 ✔ 自动区分模型
 ✔ 多路推理并行调度
 ✔ 合理分配 NPU 资源

#### 3.1.7 监控与健康检查

```bash
/healthz
```

MindIE 内置：

+　服务健康检测
+　推理失败重启
+　状态监控性能统计（吞吐、延迟）

### 3.2 MindIE的部署

