"""
文档加载与结构切分模块
对应 成品库① · 文档接入与结构切分
"""

import os
import re
from typing import List, Dict, Any, Optional

from config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


class DocumentChunk:
    """切分后的文档块"""
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"<DocumentChunk {self.metadata.get('source','')} | {len(self.text)} chars>"


def load_documents(docs_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    加载文档目录下的所有文本文件。
    实际部署时替换为 unstructured 库处理 PDF/Word/HTML。
    """
    docs_dir = docs_dir or DOCS_DIR
    documents = []

    if not os.path.isdir(docs_dir):
        print(f"[警告] 文档目录不存在: {docs_dir}")
        return documents

    for filename in os.listdir(docs_dir):
        filepath = os.path.join(docs_dir, filename)
        if not os.path.isfile(filepath):
            continue

        ext = os.path.splitext(filename)[1].lower()
        if ext not in (".txt", ".md", ".html"):
            continue

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        documents.append({
            "text": text,
            "metadata": {
                "source": filename,
                "type": ext,
            }
        })
        print(f"  [加载] {filename} ({len(text)} chars)")

    return documents


def split_documents(documents: List[Dict[str, Any]]) -> List[DocumentChunk]:
    """
    按标题层级 + 滑动窗口切分文档。
    对应简历中 chunk=512, overlap=64 的策略。
    """
    chunks = []

    for doc in documents:
        text = doc["text"]
        base_metadata = doc["metadata"]

        # 按章节切分（markdown 标题或空行）
        sections = re.split(r"\n#{1,4}\s+|\n\n+", text)

        current_chunk = ""
        current_start = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # 如果当前块 + 新段落后超出 chunk_size，先保存当前块
            if len(current_chunk) + len(section) > CHUNK_SIZE and current_chunk:
                chunks.append(DocumentChunk(
                    text=current_chunk.strip(),
                    metadata={**base_metadata, "chunk_start": current_start},
                ))
                # overlap: 保留上一块最后 overlap 字符
                overlap_text = current_chunk[-CHUNK_OVERLAP:] if len(current_chunk) > CHUNK_OVERLAP else ""
                current_chunk = overlap_text + "\n" + section
                current_start += len(current_chunk) - len(section)
            else:
                current_chunk += ("\n" + section) if current_chunk else section

        # 最后一块
        if current_chunk.strip():
            chunks.append(DocumentChunk(
                text=current_chunk.strip(),
                metadata={**base_metadata, "chunk_start": current_start},
            ))

    print(f"  [切分完成] {len(chunks)} 个 chunk (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_sample_docs(output_dir: str = None):
    """生成电信场景测试文档样本"""
    output_dir = output_dir or DOCS_DIR
    os.makedirs(output_dir, exist_ok=True)

    samples = {
        "product_manual.txt": """
5G 畅享套餐资费说明

一、套餐月费
- 129 元档：30GB 国内流量 + 500 分钟国内通话
- 199 元档：60GB 国内流量 + 1000 分钟国内通话
- 299 元档：100GB 国内流量 + 1500 分钟国内通话

二、套餐外资费
- 国内流量：5 元/GB（当月有效，自动续订）
- 国内通话：0.15 元/分钟
- 国内短信：0.1 元/条

三、合约优惠
办理 12 个月合约，每月返费 10%；
办理 24 个月合约，每月返费 15%；
办理 36 个月合约，每月返费 20%。

四、办理条件
- 个人用户凭身份证办理
- 政企用户需提供营业执照及经办人身份证
- 同一证件最多办理 5 张副卡
""",
        "fault_handbook.txt": """
宽带故障处理手册

故障一：宽带无法连接
可能原因：
1. 光猫电源未开启 — 检查光猫指示灯
2. 网线松动 — 重新插拔网线
3. 账号欠费 — 查询余额并缴费
处理步骤：
Step 1: 检查光猫 POWER 灯是否常亮
Step 2: 检查 OPTICAL 灯是否常亮（闪烁表示光路故障需报修）
Step 3: 检查 LAN 灯是否闪烁（不亮表示网线或设备问题）

故障二：网速慢
可能原因：
1. WiFi 信号干扰 — 切换到 5G 频段
2. 后台占用高 — 关闭非必要应用
3. 带宽不足 — 升级套餐
处理步骤：
Step 1: 使用有线连接测速排除 WiFi 问题
Step 2: 重启路由器
Step 3: 联系客服查询线路质量

故障三：IPTV 卡顿
可能原因：
1. 网络带宽不足
2. 机顶盒缓存过多
3. 组播配置异常
""",
        "enterprise_solution.txt": """
政企客户专线解决方案

一、产品概述
政企专线是为企业客户提供的专用数据传输通道，支持 MPLS VPN、SD-WAN 等多种接入方式。

二、产品规格
- 标准专线：10Mbps - 1000Mbps
- 精品专线：SLA 99.99%，提供双路由保护
- 跨境专线：支持香港、新加坡、日本等主要节点

三、资费标准
- 10M：800 元/月
- 50M：2000 元/月
- 100M：3500 元/月
- 1000M：15000 元/月

四、安装流程
1. 客户提交申请材料
2. 技术勘察（3 个工作日内）
3. 施工安装（勘察后 7 个工作日内）
4. 验收测试
5. 正式开通
""",
    }

    for filename, content in samples.items():
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content.strip())
        print(f"  [生成] {filename}")

    print(f"  共 {len(samples)} 个测试文档")


if __name__ == "__main__":
    print("生成测试文档...")
    build_sample_docs()
    print("\n加载文档...")
    docs = load_documents()
    print("\n切分文档...")
    chunks = split_documents(docs)
    print(f"\n共生成 {len(chunks)} 个文档块")
    for c in chunks[:3]:
        print(f"  - {c.metadata['source']}: {c.text[:60]}...")
