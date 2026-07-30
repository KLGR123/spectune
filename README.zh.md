# spectune

通过完整流水线训练你自己的 NMR 和 EI-MS 谱图解析智能体，包含数据集构建与增强、rollout、后训练（SFT 与 RL）以及端到端评估。

## 安装

```bash
cd spectune
pip install -e .                              # 仅核心包
pip install -e ".[dev]"                       # + pytest、ruff
pip install -e ".[dev,chem,mcp,reaction]"     # + rdkit、fastmcp（nmr_forward_predict）、pandas（chempile parquet）
pip install -e ".[dev,data]"                  # + pandas/pyarrow（dataloader 读取 parquet/CSV）
pip install -e ".[dev,data,classifier]"       # + sklearn/RDKit/transformers/torch/matplotlib（聚类）
```

## 凭据配置

大多数工具无需额外配置即可使用；未设置的 URL / 密钥会让对应工具返回 `status="unavailable"`，而不会直接报错。

```bash
export VOLCENGINE_WEBSEARCH_API_KEY=<key>   # web_search
export SANDBOX_FUSION_URL=<url>             # code_interpreter（沙箱后端）
export NMR_GENERATE_API_URL=<url>           # nmr_generate
export NMR_REPAIR_API_URL=<url>             # nmr_repair
export NMR_RANK_API_URL=<url>               # nmr_rerank
export NMR_PREDICT_MCP_URL=<url>            # nmr_forward_predict（需要 `pip install spectune[mcp]`）
export NMREXP_SEARCH_MCP_BASE_URL=<url>     # nmrexp_search
...
```

详见 `secrets.env.example`。

```bash
source secrets.env
```

## 测试

```bash
cd spectune
pytest -v                                   # 运行所有测试；若凭据/配置缺失，真实 API 测试将自动跳过
```

## 贡献

在提交 Pull Request 前，请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。所有 PR 必须通过自动化质量检查并经过维护者审阅。
