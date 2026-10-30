# 自信鑫的 AI 简报 · 中文起点版

> 每天早上，一份只盯 **AI / Agent** 的中文简报：每条新闻写清"发生了什么 + 产品视角为什么重要 + 原文链接"，再带一个信源可信度徽章。
> 本项目基于开源项目 [tballochi/daily-briefing](https://github.com/tballochi/daily-briefing) 改，搜索、去重、防编造的链路原样保留，只改了三处：中文主题配置、中文写作提示词、信源可信度徽章逻辑。

## 它是怎么工作的

```
你运行 python main.py --dry-run
  → 研究：大模型（Groq）自己决定搜什么
             → 调 Tavily 实时搜网（最多 6 次），跳过以前发过的文章
             → 选出最强的 3 篇
  → 写作：再调一次大模型，只根据真实原文片段写中文摘要
             （固定两段：发生了什么 / 为什么重要；链接只用搜索结果里的，绝不编）
  → 预览：终端打印纯文本简报，并生成 data/preview.html，用浏览器打开就是
    未来邮件的样子；不发送、不记录
```

三个防翻车设计，原项目就带好了，本版原样保留：

- **防重复发**：一天最多发一封，多个定时触发也不会发两遍。
- **防重复内容**：发过的文章按网址 + 标题记下来，60 天内不再选。
- **模型下线不宕机**：`config.yaml` 里有 `model` + `model_fallbacks` 备选链，主模型被厂商下线了就自动降到下一个。

## 本版改了什么（三处）

| 位置 | 改动 |
|---|---|
| `config.yaml` | 主题只留 AI / Agent 三条，删掉了原作者的航运保送主题；标题改成"自信鑫的 AI 简报"；注释全部中文 |
| `agent.py` 写作提示词 | 摘要固定两段中文："发生了什么" + "为什么重要（产品视角）"；只写片段里有的事实，不编造 |
| `agent.py` 徽章逻辑 | 新增 `source_credibility()`：只看链接域名，分三档——官方一手 / 主流媒体 / 待核实；域名表写在代码里，可自行增减 |

搜索、去重、写作、渲染、发邮件、记历史的链路一字没动。

## 快速开始：从申请免费 key 到跑出第一份简报

只需要两样东西：Python 3.11+，和两个免费密钥。**项目改好之前你什么都不用做；跑起来才需要你自己申请 key**（密钥只存在你自己电脑的 `.env` 里，不会进仓库）。

### 第 1 步：拿到代码，装依赖

```bash
git clone https://github.com/confidentxin/daily-briefing.git
cd daily-briefing
pip install -r requirements.txt   # 只有 5 个依赖
```

### 第 2 步：申请两个免费密钥（几分钟搞定）

**Groq 密钥（负责思考和写作），免费，密钥以 `gsk_` 开头：**

1. 打开 https://console.groq.com ，用 Google / GitHub 账号登录（不用绑卡）。
2. 左侧点 **API Keys** → **Create API Key**，名字随便填（比如 `daily-briefing`），点创建。
3. 弹出的密钥只显示一次，马上复制下来。

**Tavily 密钥（负责实时搜网），免费（每月 1000 积分），密钥以 `tvly-` 开头：**

1. 打开 https://tavily.com ，点右上角 **Get started** / **Sign up** 注册登录（不用绑卡）。
2. 登录后进控制台首页就能看到你的 API Key，直接复制。

### 第 3 步：把密钥填进 `.env`

```bash
cp .env.example .env
```

用任意文本编辑器打开 `.env`，只改前两行（其他三行是发邮件用的，试跑不用填）：

```
GROQ_API_KEY=gsk_你复制的密钥
TAVILY_API_KEY=tvly-你复制的密钥
```

也可以跑向导逐个验证（会当场告诉你密钥对不对）：

```bash
python main.py --setup
```

### 第 4 步：试跑，看到第一份简报

```bash
python main.py --dry-run
```

成功长这样：终端打印出 3 篇中文摘要（每篇两段 + 徽章 + 链接），并生成 `data/preview.html`，用浏览器打开就是未来邮件的样子。**只预览，不发送，不记录。** 看到这个文件，"跑通"就算达成。

## 改你自己的第一处：只改 `config.yaml`

不用碰代码，改这个文件就能换主题、换标题、换条数：

1. `topics:` 里增删改主题（英文关键词是给搜索引擎看的，搜得更全；正文一律中文）。
2. `title:` 换简报名字，邮件标题和预览标题会跟着变。
3. `num_articles:` 换每期条数。

改完再跑一次 `python main.py --dry-run`，对比 `data/preview.html` 前后差别——这就是 vibecoding 的核心循环：**改一处 → 跑一次 → 看结果 → 确认是自己改对了**。

## 常见报错排查

| 现象 | 一般原因 / 做法 |
|---|---|
| `TAVILY_API_KEY is not set` / `GROQ_API_KEY is not set` | `.env` 没建或没填。确认 `.env` 和 `main.py` 在同一个目录，且密钥前后没多空格、没带引号 |
| 密钥报错 401 | 密钥抄错了或带了多余字符。重新从官网复制一遍；Groq 密钥以 `gsk_` 开头，Tavily 以 `tvly-` 开头 |
| 429 / 限速 | Groq 免费额度是每分钟 12000 token，项目自带 85% 的节流保护。等一分钟再跑；连续出现就换个时间段 |
| 模型 404 / 下线 | Groq 定期下线模型。项目会自动按 `model_fallbacks` 降级；也可以去 https://console.groq.com/docs/deprecations 查还活着的模型，改 `config.yaml` 的 `model:` |
| dry-run 搜不到 / 只有片段 | 当天相关新闻少，或片段太薄。写作环节会只写能确认的部分，不硬编；换个时间再跑一次 |
| `pip install` 失败 | 确认 Python ≥ 3.11（`python --version`）。网络问题可重试，或换 pip 镜像源 |

跑测试（不用密钥、不用联网）：

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

## 下一步（本版不做）

本版只做到"本地 dry-run 跑出第一份中文简报"。跑通之后再做：

- **发邮件**：在 `.env` 里补齐 Gmail 的三个值（Gmail 应用专用密码，不是登录密码），跑 `python main.py --now` 真发一封。
- **定时每天自动跑**：原项目用 GitHub Actions + Secrets，部署方法见原仓库 README 的部署章节；也可以先在自己电脑上用系统定时任务每天跑 `--now`。

## 致谢

上游项目 [tballochi/daily-briefing](https://github.com/tballochi/daily-briefing)（MIT 协议），本版是它的中文起点分支。许可证见 `LICENSE`。
