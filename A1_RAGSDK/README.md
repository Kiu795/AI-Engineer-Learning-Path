# 昇腾Ascend RAGSDK 框架部署流程笔记

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

#### 3.2 MindIE的部署

+ 下载mindie镜像

  ![image-20251128161102189](https://cdn.jsdelivr.net/gh/kiu795/pic@main/img/image-20251128161102189.png)

- 启动命令示例，注意挂载模型文件夹

  ```bash
  docker run -it -d -u root -p 28080:28080 -p 28081:28081 -p 28082:28082 -p 28083:28083 -p 28084:28084 -p 28085:28085 -p 28086:28086 -p 28087:28087 \
  --security-opt seccomp=unconfined \
  --name=t1 \
  --privileged \
  --shm-size=2000m \
  --network=host \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  --entrypoint=/bin/bash \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/firmware/:/usr/local/Ascend/firmware/ \
  -v /usr/local/Ascend/toolbox:/usr/local/Ascend/toolbox \
  -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
  -v /var/log/npu/slog/:/var/log/npu/slog \
  -v /var/log/npu/profiling/:/var/log/npu/profiling \
  -v /var/log/npu/dump/:/var/log/npu/dump \
  -v /var/log/npu/:/usr/slog \
  -v /data/models/:/home/ma-user/aicc \
  33a909d3d6c1 bash
  ```

- 进入容器，设置环境变量

  ```bash
  # 配置CANN环境，默认安装在/usr/local目录下
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  # 配置模型仓环境变量
  source /usr/local/Ascend/atb-models/set_env.sh
  # MindIE
  source /usr/local/Ascend/mindie/latest/mindie-llm/set_env.sh
  source /usr/local/Ascend/mindie/latest/mindie-service/set_env.sh
  ```
  
- cd进`usr/local/Ascend/mindie/latest/mindie-service/conf/conf.json`

  修改如下字段：

  ```json
  {
      "Version" : "1.1.0",
      "LogConfig" :
      {
          "logLevel" : "Info",
          "logFileSize" : 20,
          "logFileNum" : 20,
          "logPath" : "logs/mindservice.log"
      },
  
      "ServerConfig" :
      {
          "ipAddress" : "0.0.0.0", 	#此处两个IP地址修改为0.0.0.0以支持服务公网访问
          "managementIpAddress" : "0.0.0.0",
          "port" : 1025,
          "managementPort" : 1026,
          "metricsPort" : 1027,
          "allowAllZeroIpListening" : true,
          "maxLinkNum" : 1000,
          "httpsEnabled" : false, 	#HTTPS通信安全认证，如果开启需要配置证书文件
          "fullTextEnabled" : false,
          ...
      },
      "BackendConfig" : {
          "backendName" : "mindieservice_llm_engine",
          "modelInstanceNumber" : 1,
          "npuDeviceIds" : [[0,1]], 	#这里修改为实际的设备数量
          "tokenizerProcessNumber" : 8,
          "multiNodesInferEnabled" : false,
          "multiNodesInferPort" : 1120,
          "interNodeTLSEnabled" : true,
          "interNodeTlsCaPath" : "security/grpc/ca/",
          "interNodeTlsCaFiles" : ["ca.pem"],
          "interNodeTlsCert" : "security/grpc/certs/server.pem",
          "interNodeTlsPk" : "security/grpc/keys/server.key.pem",
          "interNodeTlsPkPwd" : "security/grpc/pass/mindie_server_key_pwd.txt",
          "interNodeTlsCrlPath" : "security/grpc/certs/",
          "interNodeTlsCrlFiles" : ["server_crl.pem"],
          "interNodeKmcKsfMaster" : "tools/pmt/master/ksfa",
          "interNodeKmcKsfStandby" : "tools/pmt/standby/ksfb",
          "ModelDeployConfig" :
          {
              "maxSeqLen" : 1024,
              "maxInputTokenLen" : 1024,
              "truncation" : false,
              "ModelConfig" : [
                  {
                      "modelInstanceType" : "Standard",
                      "modelName" : "Qwen3-8B", #这里和下方修改为实际的模型名称&权重路径
                      "modelWeightPath" : "/home/ma-user/aicc/Qwen/Qwen3-8B",
                      "worldSize" : 2, 	#修改为几个NPU设备数量
                      "cpuMemSize" : 5,
                      "npuMemSize" : -1,
                      "backendType" : "atb",
                      "trustRemoteCode" : false 	#是否信任远程代码
                  }
              ]
          },
  
  
  ```

- 模型交互命令(命令行)

  ```bash
  curl -v http://223.244.40.15:1025/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d \
  '{
    "model": "Qwen3-8B",
    "temperature": 0.3,
    "max_tokens": 32, 	#生成长度
    "stream": true,
    "chat_template_kwargs":{"enable_thinking":false},
    "messages": [
      {
        "role": "user",
        "content": "你是谁？"
      }
    ]
  }'
  ```

#### 3.3 MindIE接口测试

进入[Apifox](app.apifox.com)，新建快捷请求，POST

![image-20251128163828384](https://cdn.jsdelivr.net/gh/kiu795/pic@main/img/image-20251128163828384.png)

`自动合并`支持流式传输。

## 4. RAG SDK

### 4.1 RAGSDK基础概念

mxRAG SDK是MindX SDK行业套件之一,专门面向大语言模型知识增强应用场景,提供快速构建问答系统的能力,使能基于昇腾AI处理器搭建实用可靠的平台系统,并且兼容LangChain框架。

### 4.2 部署RAGSDK

+ 下载镜像

  ![image-20251128164808544](https://cdn.jsdelivr.net/gh/kiu795/pic@main/img/image-20251128164808544.png)

+ 按照以下模板启动镜像

  ```bash
  docker run -u <user> -e ASCEND_VISIBLE_DEVICES=0 -itd --name=rag_sdk_demo --network=host \
    -v /path/to/model:/path/to/model:ro \
    <镜像名称>:<镜像tag>
  ```

+ 进入容器，配置环境

  ```bash
  bash /opt/package/install.sh 	#安装CANN
  ```

+ 设置环境变量

  ```bash
  source ~/.bashrc
  ```

+ 编译检索算子，以实现检索功能（可选，业务中涉及使用MindFaiss时才需要编译，此笔记中示例未执行该行命令）

  ```bash
  cd $MX_INDEX_INSTALL_PATH/tools/ && python3 aicpu_generate_model.py -t <chip_type> && python3 flat_generate_model.py -d <dim> -t <chip_type>  && cp op_models/* $MX_INDEX_MODELPATH
  ```

### 4.3  RAG SDK 运行

+ 部署RAGSDK

+ 部署LLM服务(本笔记中使用MindIE服务)

+ 部署Milvus服务(支持v2.5.0及以上版本)

+ 部署mis-tei embedding与reranker服务

+ 部署OCR服务(本笔记中项目暂未部署成功)

+ 图文并茂回答支持（可选，本笔记未部署）：

  若需解析docx、pdf文件中的图片并生成图文回答，需额外部署VLM模型服务（推荐模型：qwen2.5-vl-7b-instruct，[参考链接](https://www.hiascend.com/developer/ascendhub/detail/9eedc82e0c0644b2a2a9d0821ed5e7ad)）。

  > 注：长或宽小于256像素的图片因信息不足，将被自动丢弃。

#### 4.3.1 运行Demo

+ 容器内环境准备

  进入容器后执行以下命令安装依赖，创建工作目录，并准备app.py的代码([昇腾官方app.py](https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/RAGSDK/MainRepo/Samples/RagDemo/chat_with_ascend/app.py)此代码中大模型服务使用Openai接口，如果使用MindIE服务，参考这个文件:[app_mindie.py](https://github.com/Kiu795/AI-Engineer-Learning-Path/blob/main/A1_RAGSDK/app_mindie.py))：

  ```bash
  # 安装文档转换依赖（libreoffice）与中文字体
  apt-get install -y libreoffice fonts-noto-cjk
  
  # 安装Python依赖包
  pip3 install streamlit
  pip3 install mineru --trusted-host https://mirrors.aliyun.com/pypi/simple/
  pip3 install numpy==1.26.4 --trusted-host https://mirrors.aliyun.com/pypi/simple/
  
  # 创建Demo工作目录并进入
  mkdir -p /home/HwHiAiUser/workspace
  cd /home/HwHiAiUser/workspace
  
  # 编辑Demo代码文件（将仓库中的app_mindie.py内容复制到文件中）
  vim app_mindie.py
  ```

+ 启动WEB服务

  ```bash
  streamlit run app_mindie.py --server.address "0.0.0.0" --server.port 28080
  ```

+ 在PC浏览器中输入http://<服务IP>:<服务端口>访问

  ![image-20251128170734583](https://cdn.jsdelivr.net/gh/kiu795/pic@main/img/image-20251128170734583.png)

+ 正在解决的问题：
  + 实现在流程中屏蔽大模型输出中的<think>tag
  + ...
