"""生成中文路由分类增强训练数据（zh_augment.jsonl）。

背景：golden_balanced 的 1783 条中文样本全部是 R0（闲聊），导致 MLP 头学出
"中文→R0"捷径（英文任务 96% 准确，中文 R1/R2 全塌缩 R0）。本脚本按
rwkv-router 规则标签器的口径，用模板组合生成各 tier 的中文任务样本：

  R1 简单任务：单步问答/查询/翻译/计算/拼写/换算
  R2 复杂任务：代码生成/实现/重构/分析/设计/对比/总结/推导/多步
  R3 高难度：调试/报错诊断/崩溃分析/内存泄漏/死锁/风险修复

输出每行 {"text","tier"}，与 golden_balanced.jsonl 同构。

用法：uv run gen_zh_augment.py [--out data/zh_augment.jsonl] [--per-tier 300]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SEED = 42

TECH_TOPICS = [
    "闭包", "HTTP 协议", "JSON", "Python 的 GIL", "数据库索引", "LRU 缓存",
    "Docker 镜像", "REST API", "WebSocket", "正则表达式", "Git rebase",
    "微服务", "消息队列", "Redis 持久化", "TLS 握手", "DNS 解析",
    "协程和线程", "虚拟内存", "CPU 缓存", "二分查找", "哈希冲突", "进程和线程",
    "TCP 三次握手", "MVC 架构", "依赖注入", "GraphQL", "OAuth 2.0",
    "Kubernetes 的 Pod", "函数式编程", "面向对象的多态", "编译器和解释器",
]
LANGS = ["Python", "Rust", "JavaScript", "Go", "Java", "C++", "TypeScript", "Bash", "SQL", "C#"]
CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安", "南京", "东京"]
ENGLISH_WORDS = ["necessary", "occurrence", "maintenance", "embarrassment", "conscience", "rhythm"]
ERRORS = [
    "segfault", "段错误", "panic", "空指针异常", "NullPointerException",
    "Segmentation fault", "stack overflow", "栈溢出", "内存越界",
    "ConcurrentModificationException", "OOM", "out of memory", "死锁",
]
GENERIC_SUBJECTS = ["量子计算", "相对论", "区块链", "光合作用", "板块构造", "人工智能", "黑洞", "疫苗原理"]
NUMS = [(3, 5), (12, 47), (123, 456), (88, 91), (7, 13), (256, 128), (999, 111), (45, 67)]

# ---------- R1：简单任务（单步操作/问答/查询/翻译/计算） ----------

R1_TEMPLATES = [
    "什么是{t}？",
    "{t}是什么意思？",
    "简单介绍一下{t}",
    "{t}是哪一年提出的？",
    "帮我查一下{t}的定义",
    "{t}主要用来做什么？",
    "{t}和{t2}有什么区别？",
    "把'{zh}'翻译成英文",
    "{zh}用英语怎么说？",
    "'{zh}'的英文翻译是什么？",
    "{a}+{b}等于多少？",
    "计算 {a} 乘以 {b}",
    "{a} 除以 {b} 是多少？",
    "{w} 怎么拼写？",
    "{c}今天天气怎么样？",
    "{n} 美元换算成人民币是多少？",
    "{g}的首都是哪里？",
    "{g}是谁提出的？",
    "现在几点了？",
    "{t}的英文缩写是什么？",
    "怎么注册{t}账号？",
    "使用{t}需要付费吗？",
    "{t}支持哪些平台？",
    "如何开启{t}的调试模式？",
    "{t}的默认端口是多少？",
]


def gen_r1(rng: random.Random) -> list[str]:
    out = []
    zh_pool = [
        "今天天气真好", "谢谢你帮我", "这个主意不错", "我很喜欢这个方案",
        "祝你生日快乐", "周末过得怎么样", "最近工作顺利吗", "这道菜真好吃",
        "这部电影很精彩", "路上注意安全",
    ]
    for i, tpl in enumerate(R1_TEMPLATES):
        for j in range(16):
            t = TECH_TOPICS[(i + j * 3) % len(TECH_TOPICS)]
            t2 = TECH_TOPICS[(i + j * 3 + 7) % len(TECH_TOPICS)]
            g = GENERIC_SUBJECTS[(i + j) % len(GENERIC_SUBJECTS)]
            a, b = NUMS[(i + j) % len(NUMS)]
            w = ENGLISH_WORDS[(i + j) % len(ENGLISH_WORDS)]
            c = CITIES[(i + j) % len(CITIES)]
            zh = zh_pool[j % len(zh_pool)]
            out.append(
                tpl.format(t=t, t2=t2, g=g, a=a, b=b, w=w, c=c, zh=zh, n=(i + j * 13) % 500)
            )
    return out


# ---------- R2：复杂任务（代码生成/实现/重构/分析/设计/多步推理） ----------

R2_TEMPLATES = [
    "写一个{lang}函数，用于解析{t}",
    "帮我用{lang}实现一个{t}，要求支持并发访问",
    "实现一个{lang}类，支持 get 和 put 操作，容量可配置",
    "帮我重构这个{t}模块的错误处理逻辑，统一异常体系",
    "分析这段{lang}代码的性能瓶颈，并给出优化建议",
    "设计一个{t}，要求支持水平扩展",
    "对比{t}和{t2}的实现原理，总结各自的适用场景",
    "为这个{lang}函数编写完整的单元测试，覆盖边界情况",
    "写一个{lang}脚本，批量重命名目录下所有文件",
    "推导{t}的平均时间复杂度，给出完整证明过程",
    "总结一下这篇关于{t}的长文核心观点，分条列出",
    "帮我优化这条关于{t}的 SQL 查询，当前执行要 8 秒",
    "用{lang}实现一个简单的解释器，支持四则运算",
    "把这个{lang}项目从 {t2} 迁移到 {t}",
    "帮我画一下{t}系统的架构图，并解释各组件职责",
    "基于{t}业务设计数据库表结构：用户、订单、商品，要求支持高并发",
    "写一个{lang}爬虫，抓取{t}相关数据并入库",
    "用{lang}实现一个支持撤销/重做的{t}编辑器核心逻辑",
    "多步推理：给定一组{t}数据，计算复购率并分析影响因素",
    "帮我审查这段{lang}代码，指出潜在问题并给出改进方案",
    "写一个{lang}命令行工具，支持 {t} 的增删改查",
    "用{lang}实现{t}，要有完整注释和使用示例",
    "帮我写技术方案文档：如何给现有系统加上{t}支持",
    "用{lang}实现一个简单的{t}规则引擎，支持条件组合和优先级",
]


def gen_r2(rng: random.Random) -> list[str]:
    out = []
    for i, tpl in enumerate(R2_TEMPLATES):
        for j in range(16):
            lang = LANGS[(i + j) % len(LANGS)]
            t = TECH_TOPICS[(i + j * 5) % len(TECH_TOPICS)]
            t2 = TECH_TOPICS[(i + j * 5 + 11) % len(TECH_TOPICS)]
            out.append(tpl.format(lang=lang, t=t, t2=t2))
    # 带代码块的任务请求
    code_reqs = [
        "帮我看看这段代码哪里可以优化：\n```python\ndef find_duplicates(items):\n    result = []\n    for i in range(len(items)):\n        for j in range(i + 1, len(items)):\n            if items[i] == items[j] and items[i] not in result:\n                result.append(items[i])\n    return result\n```",
        "把这段代码改成异步版本：\n```javascript\nasync function fetchAll(urls) {\n  const results = [];\n  for (const u of urls) {\n    results.push(await fetch(u));\n  }\n  return results;\n}\n```",
        "解释这段 Rust 代码的生命周期问题并修复：\n```rust\nfn longest<'a>(x: &'a str, y: &'a str) -> &'a str {\n    if x.len() > y.len() { x } else { y }\n}\n```",
        "为这个函数写单元测试：\n```go\nfunc ParseDuration(s string) (time.Duration, error) {\n    // ...\n}\n```",
    ]
    out.extend(code_reqs)
    return out


# ---------- R3：高难度（调试/诊断/崩溃/泄漏/死锁/风险） ----------

R3_TEMPLATES = [
    "为什么这段代码会抛出{err}",
    "帮我调试这个{err}，找出根因",
    "程序启动时报 {err}，帮我排查原因",
    "帮我分析这段{lang}代码的内存泄漏问题",
    "这段{lang}代码不工作，帮我看看哪里出错了",
    "编译{lang}项目时出现{err}，不知道哪里的问题",
    "修复这个{lang}并发场景下的{err}问题",
    "服务每隔几小时就{err}，帮忙定位根因",
    "上线后{t}相关接口偶发超时，帮我排查是哪里的问题",
    "{t}连接池耗尽导致服务不可用，帮我分析原因并修复",
    "这段{lang}代码在高并发下数据错乱，帮我找出竞态条件",
    "内存占用持续增长不释放，怀疑{t}有泄漏，帮我分析",
    "帮我诊断这个{t}崩溃堆栈，定位到具体代码行",
    "修复{t}模块的这个安全漏洞：SQL 注入风险",
    "生产环境{t}数据不一致，帮我排查事务问题",
    "这个{lang}递归实现导致{err}，帮我修复并解释原因",
    "升级{t}依赖后测试全挂了，帮我分析 API 变更影响并修复",
    "{lang}多进程共享文件出现内容错乱，帮我定位问题",
]


def gen_r3(rng: random.Random) -> list[str]:
    out = []
    for i, tpl in enumerate(R3_TEMPLATES):
        for j in range(16):
            err = ERRORS[(i + j) % len(ERRORS)]
            lang = LANGS[(i + j * 3) % len(LANGS)]
            t = TECH_TOPICS[(i + j * 7) % len(TECH_TOPICS)]
            out.append(tpl.format(err=err, lang=lang, t=t))
    # 带堆栈/代码的报错请求
    stack_reqs = [
        "帮我看看这个报错怎么修：\n```\nTraceback (most recent call last):\n  File \"app.py\", line 42, in handler\n    return process(data[\"key\"])\nKeyError: 'key'\n```",
        "程序崩溃了，帮我分析：\n```\npanic: runtime error: invalid memory address or nil pointer dereference\ngoroutine 1 [running]:\nmain.process(0x0)\n```",
        "Java 服务抛异常，帮我定位：\n```\nException in thread \"main\" java.lang.NullPointerException\n\tat com.app.Service.handle(Service.java:87)\n```",
        "这段代码偶发崩溃，帮我找出原因：\n```cpp\nint* p = nullptr;\nif (cond) { p = new int(5); }\n*p = 10;  // cond 为 false 时崩溃\n```",
        "为什么会产生死锁，帮我修复：\n```python\nlock_a.acquire()\nlock_b.acquire()\n# 另一个线程反序获取\n```",
    ]
    out.extend(stack_reqs)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/zh_augment.jsonl")
    parser.add_argument("--per-tier", type=int, default=300)
    args = parser.parse_args()

    rng = random.Random(SEED)
    pools = {
        1: gen_r1(rng),
        2: gen_r2(rng),
        3: gen_r3(rng),
    }

    out_path = Path(args.out)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for tier, pool in pools.items():
            # 去重后取前 per-tier 条（模板 × 变体天然超过该数）
            seen: set[str] = set()
            unique = [s for s in pool if not (s in seen or seen.add(s))]
            rng.shuffle(unique)
            for text in unique[: args.per_tier]:
                f.write(json.dumps({"text": text, "tier": tier}, ensure_ascii=False) + "\n")
                count += 1

    print(f"generated {count} samples -> {out_path}")
    # 语言分布自检
    tiers = [0, 0, 0, 0]
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            tiers[json.loads(line)["tier"]] += 1
    print(f"tier counts: {tiers}")


if __name__ == "__main__":
    main()
