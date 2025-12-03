import agent_framework
from agent_framework import AgenticRAG
from agent_framework import R1Searcher

"""
MindIE Searcher - 为 MindIE 推理引擎设计的 AgenticRAG 适配器
支持 OpenAI 兼容 API 和潜在的原生 MindIE SDK
"""

from typing import List, Optional, Dict, Any
import pandas as pd
from openai import OpenAI
import os


class MindIESearcher(AgenticRAG):
    """
    MindIE 后端的 AgenticRAG 框架适配器。
    使用 OpenAI 兼容的 API 与 MindIE 服务通信。
    
    使用示例:
        retriever = pt.BatchRetrieve(index, wmodel="BM25")
        
        mindie_searcher = MindIESearcher(
            retriever=retriever,
            base_url="http://localhost:8000/v1",  # MindIE 服务端点
            model_name="qwen",
            temperature=0.7,
            top_k=8,
            max_turn=6,
            max_tokens=512,
            prompt_type='v1'
        )
        
        results = mindie_searcher.transform(test_queries)
    """
    
    def __init__(self, 
                 retriever,
                 base_url: str = "http://localhost:8000/v1",
                 api_key: str = "dummy",  # MindIE 本地服务可能不需要真实密钥
                 model_name: str = "Qwen3-8B",
                 generator=None,
                 temperature=0.7,
                 top_k=8,
                 top_p=0.95,
                 max_turn=6,
                 max_tokens=512,
                 prompt_type='v1',
                 timeout=60,
                 verbose=True,
                 enable_thinking=True,  # MindIE 特定参数：是否启用思考模式
                 stream=True,           # MindIE 特定参数：是否使用流式输出
                 **kwargs):
        """
        初始化 MindIE Searcher。
        
        参数:
            retriever: PyTerrier 检索器，用于文档搜索
            base_url: MindIE 服务 URL（OpenAI 兼容端点）
            api_key: API 密钥（本地服务可以使用虚拟值）
            model_name: API 调用中使用的模型名称（如 "Qwen3-8B"）
            temperature: 采样温度
            top_k: 检索的文档数量
            top_p: 核采样参数
            max_turn: 最大推理轮次
            max_tokens: 每次生成的最大 token 数
            prompt_type: 提示模板版本 ('v0', 'v1', 'v2', 'v3')
            timeout: 请求超时时间（秒）
            verbose: 启用详细日志
            enable_thinking: MindIE 特定参数，是否启用思考模式
            stream: 是否使用流式输出（当前实现为非流式）
        """
        super().__init__(
            retriever=retriever,
            generator=None,  # 我们内部处理生成
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_turn=max_turn,
            max_tokens=max_tokens,
            model_id=model_name,
            prompt=self.get_prompt(prompt_type),
            **kwargs
        )
        
        self.base_url = base_url
        self.model_name = model_name
        self.prompt_type = prompt_type
        self.verbose = verbose
        self.timeout = timeout
        self.enable_thinking = enable_thinking
        self.stream = stream
        
        # 为 MindIE 初始化 OpenAI 客户端
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout
        )
        
        # 根据提示类型设置搜索标签
        if prompt_type in ['v0', 'v1', 'v2', 'v3']:
            self.start_search_tag = "<|begin_of_query|>"
            self.end_search_tag = "<|end_of_query|>"
            self.start_results_tag = "<|begin_of_documents|>"
            self.end_results_tag = "<|end_of_documents|>"
        else:
            # 回退到 SearchR1 风格的标签
            self.start_search_tag = "<search>"
            self.end_search_tag = "</search>"
            self.start_results_tag = "<information>"
            self.end_results_tag = "</information>"
        
        if self.verbose:
            print(f"[MindIESearcher] 已初始化，端点: {base_url}")
            print(f"[MindIESearcher] 模型: {model_name}, 提示类型: {prompt_type}")
    
    def get_prompt(self, prompt_type: str) -> str:
        """根据类型获取提示模板。"""
        if prompt_type == 'v0':
            return """用户提出一个问题，助手解决它。
助手首先在脑海中思考推理过程，然后为用户提供最终答案。
推理过程和最终答案的输出格式分别包含在 <think> </think> 和 <answer> </answer> 标签中，即 "<think> 推理过程 </think>\n\n<answer> 最终答案 </answer>"。
在思考过程中，如有必要，助手可以通过 "<|begin_of_query|> 搜索查询（仅关键词） <|end_of_query|>" 格式搜索不确定的知识。**一个查询必须只涉及单个三元组**。
然后，系统将以 "<|begin_of_documents|> ...搜索结果... <|end_of_documents|>" 格式为助手提供有用信息。\n\n用户:{question}\n助手: <think>"""

        elif prompt_type == 'v1':
            return """用户提出一个问题，助手解决它。
仅使用这些标签: <think>...</think>, <|begin_of_query|>...<|end_of_query|>, <|begin_of_documents|>...<|end_of_documents|>, <answer>...</answer>。
一般协议:
1) 在 <think> 中，如需要则分解问题，并决定缺少什么信息。
2) 需要外部知识时，输出恰好一行:
   <|begin_of_query|> 关键词1\t关键词2\t... <|end_of_query|>
   - 包含核心实体/主体和基本属性/约束关键词。
   - 在有帮助时添加常见别名/同义词（英文和/或中文）。
   - 在 <|end_of_query|> 后立即停止。在提供 <|begin_of_documents|> 之前不要输出任何其他内容。
3) 在我返回 <|begin_of_documents|> ... <|end_of_documents|> 后，恢复 <think> 以提取所需事实:
   - 优先使用直接支持需求的明确陈述。
   - 如果证据不足或偏离主题，改进关键词并再次搜索。
4) 只有当 <|begin_of_documents|> ... <|end_of_documents|> 中有明确的支持证据时，才输出:
   <answer> 最终答案 </answer>

输出规则:
- 保持 <think> 简洁；不要在标签外透露思维链。
- 在从 <|begin_of_documents|> 找到证据之前不要输出 <answer>。
- 如果经过多次搜索仍不确定，继续搜索；不要猜测。
用户:{question}
助手: <think>"""

        elif prompt_type == 'v2':
            return """用户提出一个问题，助手解决它。
助手首先在脑海中思考推理过程，然后为用户提供最终答案。
推理过程和最终答案的输出格式分别包含在 <think> </think> 和 <answer> </answer> 标签中。
在思考过程中，**助手可以执行搜索**，如有必要搜索不确定的知识，格式为 "<|begin_of_query|> 搜索查询（仅列出用 "\t" 分隔的关键词，而不是完整句子，如 **"关键词_1 \t 关键词_2 \t..."**)<|end_of_query|>"。**一个查询必须只涉及单个三元组**。
然后，搜索系统将以 "<|begin_of_documents|> ...搜索结果... <|end_of_documents|>" 格式为助手提供检索信息。

用户:{question}
助手: <think>"""

        elif prompt_type == 'v3':
            return """用户提出一个**判断问题**，助手解决它。助手首先在脑海中思考推理过程，然后为用户提供最终答案。推理过程和最终答案的输出格式分别包含在 <think> </think> 和 <answer> </answer> 标签中。在思考过程中，如有必要，助手可以通过 "<|begin_of_query|> 搜索查询（仅关键词） <|end_of_query|>" 格式搜索不确定的知识。然后，系统将以 "<|begin_of_documents|> ...搜索结果... <|end_of_documents|>" 格式为助手提供有用信息。最终答案**必须是 yes 或 no**。\n\n用户:{question}\n助手: <think>"""
        
        else:  # 默认/SearchR1 风格
            return """回答给定的问题。\
你必须首先在 <think> 和 </think> 内进行推理，每次获得新信息时都要这样做。\
推理后，如果你发现缺少某些知识，可以通过 <search> 查询 </search> 调用搜索引擎，它将在 <information> 和 </information> 之间返回顶部搜索结果。\
你可以搜索任意次数。\
如果你发现不需要进一步的外部知识，可以直接在 <answer> 和 </answer> 内提供答案，无需详细说明。例如，<answer> 北京 </answer>。问题: 中国的首都是什么？
用户:{question}
助手: <think>"""
    
    def generate(self, contexts: List[str]) -> List[str]:
        """
        批量生成：使用 OpenAI 客户端调用 MindIE 服务。
        支持流式和非流式两种模式。
        
        参数:
            contexts: 输入上下文列表（每个都是完整的对话历史）
            
        返回:
            生成的文本列表
        """
        texts: List[str] = []
        
        for context in contexts:
            try:
                # 构建请求参数
                request_params = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": context}
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "stream": self.stream,
                    # 设置停止序列以在搜索标签处停止
                    "stop": [
                        self.end_search_tag,
                        "</answer>",
                        "<|im_end|>",
                        "<|endoftext|>"
                    ]
                }
                
                # 添加 MindIE 特定参数（通过 extra_body 传递）
                request_params["extra_body"] = {
                    "chat_template_kwargs": {
                        "enable_thinking": self.enable_thinking
                    }
                }
                
                # 根据是否流式选择不同的处理方式
                if self.stream:
                    generated_text = self._generate_stream(**request_params)
                else:
                    response = self.client.chat.completions.create(**request_params)
                    generated_text = response.choices[0].message.content or ""
                
                # 后处理：确保搜索标签完整
                normalized = generated_text.strip()
                last_begin_query = normalized.rfind(self.start_search_tag)
                last_answer = normalized.rfind("<answer>")
                
                # 如果有未闭合的搜索查询标签，添加结束标签
                if last_begin_query != -1 and (last_answer == -1 or last_begin_query > last_answer):
                    if normalized.rfind(self.end_search_tag, last_begin_query) == -1:
                        normalized = normalized.rstrip() + f" {self.end_search_tag}"
                # 如果有未闭合的答案标签，添加结束标签
                elif last_answer != -1:
                    if normalized.rfind("</answer>", last_answer) == -1:
                        normalized = normalized.rstrip() + " </answer>"
                
                texts.append(normalized)
                
                if self.verbose:
                    print(f"[MindIESearcher] 生成了 {len(normalized)} 个字符")
                    
            except Exception as e:
                error_msg = f"生成失败: {str(e)}"
                if self.verbose:
                    print(f"[MindIESearcher] 错误: {error_msg}")
                # 返回空字符串或错误消息，以便流程继续
                texts.append("")
        
        return texts
    
    def _generate_stream(self, **request_params) -> str:
        """
        流式生成：实时累积内容并在检测到停止标记时提前终止。
        
        返回:
            累积的完整生成文本
        """
        accumulated_text = ""
        
        # 调用流式 API
        stream = self.client.chat.completions.create(**request_params)
        
        # 逐块累积内容
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    accumulated_text += delta.content
                    
                    if self.verbose:
                        print(delta.content, end="", flush=True)
                    
                    # 检查是否遇到停止标记（提前终止）
                    # 检查搜索标签结束
                    if self.end_search_tag in accumulated_text:
                        if self.verbose:
                            print(f"\n[MindIESearcher] 检测到搜索标签结束: {self.end_search_tag}")
                        break
                    
                    # 检查答案标签结束
                    if "</answer>" in accumulated_text:
                        if self.verbose:
                            print("\n[MindIESearcher] 检测到答案标签结束")
                        break
                
                # 检查是否完成
                if chunk.choices[0].finish_reason:
                    if self.verbose:
                        print(f"\n[MindIESearcher] 流式生成完成: {chunk.choices[0].finish_reason}")
                    break
        
        if self.verbose:
            print()  # 换行
        
        return accumulated_text


class MindIESearcherHTTP(AgenticRAG):
    """
    使用原始 HTTP 请求的 MindIE Searcher 变体（不依赖 OpenAI 库）。
    适用于需要更多控制或 OpenAI 库不可用的场景。
    """
    
    def __init__(self, 
                 retriever,
                 base_url: str = "http://localhost:8000",
                 model_name: str = "Qwen3-8B",
                 temperature=0.7,
                 top_k=8,
                 top_p=0.95,
                 max_turn=6,
                 max_tokens=512,
                 prompt_type='v1',
                 timeout=60,
                 verbose=True,
                 enable_thinking=False,
                 stream=False,
                 **kwargs):
        """
        使用 HTTP 请求初始化 MindIE Searcher。
        
        参数同 MindIESearcher，但使用原始 HTTP 请求而非 OpenAI 客户端。
        """
        super().__init__(
            retriever=retriever,
            generator=None,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_turn=max_turn,
            max_tokens=max_tokens,
            model_id=model_name,
            prompt=MindIESearcher.get_prompt(self, prompt_type),
            **kwargs
        )
        
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.prompt_type = prompt_type
        self.verbose = verbose
        self.timeout = timeout
        self.enable_thinking = enable_thinking
        self.stream = stream
        
        # 根据提示类型设置搜索标签
        if prompt_type in ['v0', 'v1', 'v2', 'v3']:
            self.start_search_tag = "<|begin_of_query|>"
            self.end_search_tag = "<|end_of_query|>"
            self.start_results_tag = "<|begin_of_documents|>"
            self.end_results_tag = "<|end_of_documents|>"
        else:
            self.start_search_tag = "<search>"
            self.end_search_tag = "</search>"
            self.start_results_tag = "<information>"
            self.end_results_tag = "</information>"
        
        if self.verbose:
            print(f"[MindIESearcherHTTP] 已初始化，端点: {base_url}")
    
    def generate(self, contexts: List[str]) -> List[str]:
        """
        使用原始 HTTP 请求批量生成。
        支持流式和非流式两种模式。
        
        参数:
            contexts: 输入上下文列表
            
        返回:
            生成的文本列表
        """
        import requests
        import json
        
        texts: List[str] = []
        
        for context in contexts:
            try:
                # 构建请求体，包含 MindIE 特定参数
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": context}],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "stream": self.stream,
                    "chat_template_kwargs": {
                        "enable_thinking": self.enable_thinking
                    },
                    "stop": [
                        self.end_search_tag,
                        "</answer>",
                        "<|im_end|>",
                        "<|endoftext|>"
                    ]
                }
                
                # 根据是否流式选择不同的处理方式
                if self.stream:
                    generated_text = self._generate_stream_http(payload)
                else:
                    response = requests.post(
                        f"{self.base_url}/v1/chat/completions",
                        headers={"Content-Type": "application/json"},
                        json=payload,
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        generated_text = result['choices'][0]['message']['content']
                    else:
                        if self.verbose:
                            print(f"[MindIESearcherHTTP] HTTP 错误 {response.status_code}: {response.text}")
                        generated_text = ""
                
                # 后处理（与 MindIESearcher 相同）
                normalized = generated_text.strip()
                last_begin_query = normalized.rfind(self.start_search_tag)
                last_answer = normalized.rfind("<answer>")
                
                if last_begin_query != -1 and (last_answer == -1 or last_begin_query > last_answer):
                    if normalized.rfind(self.end_search_tag, last_begin_query) == -1:
                        normalized = normalized.rstrip() + f" {self.end_search_tag}"
                elif last_answer != -1:
                    if normalized.rfind("</answer>", last_answer) == -1:
                        normalized = normalized.rstrip() + " </answer>"
                
                texts.append(normalized)
                    
            except Exception as e:
                if self.verbose:
                    print(f"[MindIESearcherHTTP] 请求失败: {str(e)}")
                texts.append("")
        
        return texts
    
    def _generate_stream_http(self, payload: dict) -> str:
        """
        使用原始 HTTP 请求的流式生成。
        
        参数:
            payload: 请求体字典
            
        返回:
            累积的完整生成文本
        """
        import requests
        import json
        
        accumulated_text = ""
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                stream=True,  # 启用流式接收
                timeout=self.timeout
            )
            
            # 逐行读取 SSE 数据
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    
                    # SSE 格式：每行以 "data: " 开头
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # 去掉 "data: " 前缀
                        
                        # 检查是否结束
                        if data_str == '[DONE]':
                            if self.verbose:
                                print("\n[MindIESearcherHTTP] 流式生成完成: [DONE]")
                            break
                        
                        try:
                            chunk = json.loads(data_str)
                            
                            # 提取增量内容
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                
                                if content:
                                    accumulated_text += content
                                    
                                    if self.verbose:
                                        print(content, end="", flush=True)
                                    
                                    # 检查是否遇到停止标记（提前终止）
                                    if self.end_search_tag in accumulated_text:
                                        if self.verbose:
                                            print(f"\n[MindIESearcherHTTP] 检测到搜索标签结束")
                                        break
                                    
                                    if "</answer>" in accumulated_text:
                                        if self.verbose:
                                            print("\n[MindIESearcherHTTP] 检测到答案标签结束")
                                        break
                                
                                # 检查完成原因
                                finish_reason = chunk['choices'][0].get('finish_reason')
                                if finish_reason:
                                    if self.verbose:
                                        print(f"\n[MindIESearcherHTTP] 流式生成完成: {finish_reason}")
                                    break
                        
                        except json.JSONDecodeError as e:
                            if self.verbose:
                                print(f"\n[MindIESearcherHTTP] JSON 解析错误: {e}")
                            continue
            
            if self.verbose:
                print()  # 换行
                
        except Exception as e:
            if self.verbose:
                print(f"\n[MindIESearcherHTTP] 流式请求错误: {str(e)}")
        
        return accumulated_text