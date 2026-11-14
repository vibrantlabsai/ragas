<h1 align="center">
  <img style="vertical-align:middle" height="200"
  src="https://raw.githubusercontent.com/vibrantlabsai/ragas/main/docs/_static/imgs/logo.png">
</h1>
<p align="center">
  <i>为你的大模型应用评估全面提效赋能 🚀</i>
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<p align="center">
    <a href="https://github.com/vibrantlabsai/ragas/releases">
        <img alt="Latest release" src="https://img.shields.io/github/release/vibrantlabsai/ragas.svg">
    </a>
    <a href="https://www.python.org/">
        <img alt="Made with Python" src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple">
    </a>
    <a href="https://github.com/vibrantlabsai/ragas/blob/master/LICENSE">
        <img alt="License Apache-2.0" src="https://img.shields.io/github/license/vibrantlabsai/ragas.svg?color=green">
    </a>
    <a href="https://pypi.org/project/ragas/">
        <img alt="Ragas Downloads per month" src="https://static.pepy.tech/badge/ragas/month">
    </a>
    <a href="https://discord.gg/5djav8GGNZ">
        <img alt="Join Ragas community on Discord" src="https://img.shields.io/discord/1119637219561451644">
    </a>
    <a target="_blank" href="https://deepwiki.com/vibrantlabsai/ragas">
      <img 
        src="https://devin.ai/assets/deepwiki-badge.png" 
        alt="Ask DeepWiki.com" 
        height="20" 
      />
    </a>
</p>

<h4 align="center">
    <p>
        <a href="https://docs.ragas.io/">官方文档</a> |
        <a href="#fire-快速上手">快速上手</a> |
        <a href="https://discord.gg/5djav8GGNZ">加入 Discord</a> |
        <a href="https://blog.ragas.io/">官方博客</a> |
        <a href="https://newsletter.ragas.io/">时事通讯</a> |
        <a href="https://www.ragas.io/careers">招贤纳士</a>
    <p>
</h4>

为大语言模型应用量身打造的客观度量指标、智能测试集生成与数据驱动洞察。

Ragas 是用于评估与优化大语言模型（LLM）应用的终极工具套件。告别耗时、主观的人工打分，拥抱以数据为驱动、高效且标准化的自动化评测工作流。
还没有准备好测试数据集？我们还支持自动生成与生产环境场景紧密对齐的合成测试集。

## 核心特性

- 🎯 **客观度量指标 (Objective Metrics)**：结合基于 LLM 的智能评分机制与传统统计评估方法，精准评测您的 LLM 应用。
- 🧪 **测试数据生成 (Test Data Generation)**：自动化创建覆盖广泛业务场景的综合测试数据集。
- 🔗 **无缝生态集成 (Seamless Integrations)**：与 LangChain 等主流 LLM 开发框架及各类主流可观测性工具开箱即用、无缝协同。
- 📊 **构建数据反馈闭环 (Build feedback loops)**：充分利用生产环境真实数据，持续驱动大模型应用迭代优化。

## :shield: 安装指南

通过 PyPI 安装：

```bash
pip install ragas
```

或者直接从 GitHub 源码安装最新版本：

```bash
pip install git+https://github.com/vibrantlabsai/ragas
```

## :fire: 快速上手

### 克隆完整示例工程

最快的上手方式是使用 `ragas quickstart` 命令行工具：

```bash
# 列出所有可用的项目模板
ragas quickstart

# 快速创建一个 RAG 评估工程
ragas quickstart rag_eval

# 指定工程生成的目标目录路径
ragas quickstart rag_eval -o ./my-project
```

当前可用模板：
- `rag_eval` - 检索增强生成 (RAG) 系统评估

即将推出：
- `agent_evals` - AI 智能体 (AI Agents) 综合评估
- `benchmark_llm` - 大语言模型基准对比与评测
- `prompt_evals` - Prompt 提示词变体效果评估
- `workflow_eval` - 复杂业务工作流综合评估

### 评测你的 LLM 应用

`ragas` 内置了针对常见评测场景的开箱即用指标。例如，利用维度评价器（Aspect Critique）中的 `DiscreteMetric`，可以对模型输出的任意维度进行精确评估：

```python
import asyncio
from openai import AsyncOpenAI
from ragas.metrics import DiscreteMetric
from ragas.llms import llm_factory

# 1. 初始化大模型客户端
client = AsyncOpenAI()
llm = llm_factory("gpt-4o", client=client)

# 2. 创建自定义的维度评估器
metric = DiscreteMetric(
    name="summary_accuracy",
    allowed_values=["accurate", "inaccurate"],
    prompt="""Evaluate if the summary is accurate and captures key information.

Response: {response}

Answer with only 'accurate' or 'inaccurate'."""
)

# 3. 对应用的生成结果进行打分评测
async def main():
    score = await metric.ascore(
        llm=llm,
        response="The summary of the text is..."
    )
    print(f"Score: {score.value}")  # 输出 'accurate' 或 'inaccurate'
    print(f"Reason: {score.reason}")


if __name__ == "__main__":
    asyncio.run(main())
```

> **说明**：运行前请确保已配置 `OPENAI_API_KEY` 环境变量。

查阅完整的[快速入门指南 (Quickstart Guide)](https://docs.ragas.io/en/latest/getstarted/quickstart)。

## 希望通过评测技术提升您的 AI 应用质量？

在过去的两年中，我们见证并协助了众多团队利用评测驱动的方法显著改进其 AI 应用。如果您希望借助自动化评测优化并规模化扩展您的 AI 业务：

🔗 欢迎[预约交流日程](https://cal.com/team/vibrantlabs/app) 或发送邮件至：[founders@vibrantlabs.com](mailto:founders@vibrantlabs.com)。

## 🫂 开发者社区

想要更深入地参与 Ragas 建设？欢迎加入我们的 [Discord 官方社区](https://discord.gg/5qGUJ6mh7C)。在这里，大家一起深度探讨大模型、检索系统、生产环境落地难题以及前沿 AI 技术的方方面面。

## 参与贡献

```yml
+----------------------------------------------------------------------------+
|     +----------------------------------------------------------------+     |
|     | 开发者：使用 `ragas` 进行构建的用户。                          |     |
|     | （你的代码中某处包含 `import ragas`）                          |     |
|     |     +----------------------------------------------------+     |     |
|     |     | 贡献者：让 `ragas` 变得更加完善的用户。            |     |     |
|     |     | （你向本仓库提交了 Pull Request）                  |     |     |
|     |     +----------------------------------------------------+     |     |
|     +----------------------------------------------------------------+     |
+----------------------------------------------------------------------------+
```

我们由衷欢迎来自社区的每一份贡献！无论是修复 Bug、添加新功能还是改进文档，您的参与都至关重要：

1. Fork 本仓库到您的个人 GitHub 账号
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的修改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支到您的远端仓库 (`git push origin feature/AmazingFeature`)
5. 发起 Pull Request

## 🔍 透明遥测说明 (Open Analytics)

在 Ragas，我们始终坚持公开与透明原则。我们仅收集极少数去标识化、完全匿名的使用数据，用于改进产品体验并指导后续研发投入方向。

✅ 绝不收集任何个人或企业身份识别信息

✅ 遥测数据收集源码完全公开开源：[查看源码](./src/ragas/_analytics.py)

✅ 遥测统计汇总数据面向公众完全开放：[查看汇总数据](https://github.com/vibrantlabsai/ragas/issues/49)

如需退出遥测，只需将环境变量 `RAGAS_DO_NOT_TRACK` 设置为 `true` 即可。

### 引用本项目

```bibtex
@misc{ragas2024,
  author       = {VibrantLabs},
  title        = {Ragas: Supercharge Your LLM Application Evaluations},
  year         = {2024},
  howpublished = {\url{https://github.com/vibrantlabsai/ragas}},
}
```

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年9月13日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
