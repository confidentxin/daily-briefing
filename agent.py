"""AI agent for the daily tech briefing.

Two phases, both driven by a Groq-hosted LLM (see `model` in config.yaml):

1. Research agent (tool-using loop): the LLM decides which searches to run, judges
   the results, skips articles already sent on previous days, avoids index/nav
   pages, and selects the 3 most important distinct stories.
2. Synthesis: a focused call writes the factual summaries grounded ONLY in the real
   source snippets, plus a word of the day and a quote of the day.

Separating research from writing keeps summaries real and source-grounded, and the
final URLs come from actual search results (the LLM never fabricates a link). The
loop is paced to respect the Groq free-tier limit of 12,000 tokens/minute.
"""

import os
import re
import json
import time
import logging
import urllib.error
import urllib.request
from datetime import datetime
from urllib.parse import urlparse

from groq import Groq
from tavily import TavilyClient

import config as briefing_config
import history

logger = logging.getLogger("briefing.agent")

# Free-tier guardrails ------------------------------------------------------
TPM_LIMIT = 12000          # Groq free-tier tokens-per-minute ceiling
TPM_SAFETY = 0.85          # only use 85% of the budget to stay clear of 429s
MAX_STEPS = 10             # hard cap on agent loop iterations
MAX_SEARCHES = 6           # hard cap on Tavily searches per run (quota friendly)

# Token budget for the writing call. Reasoning models spend most of it thinking before
# they emit anything: gpt-oss-120b used ~1000 reasoning tokens on a 3-article briefing,
# so the old 1500 cap left the JSON truncated and Groq's validator rejected it
# ("max completion tokens reached before generating a valid document").
SYNTHESIS_MAX_TOKENS = 4000

# User preferences (topics, focus theme, article count) live in config.yaml so the
# briefing is personalisable without touching the code. See config.py.
CFG = briefing_config.load()
TARGET_ARTICLES = CFG.num_articles   # how many stories the briefing contains
FOCUS = CFG.focus                    # optional always-include theme, or None
FOCUS_KEYWORDS = FOCUS["keywords"] if FOCUS else ()

# Minimum number of focus articles to guarantee in the briefing
MIN_FOCUS_ARTICLES = 2 if FOCUS else 0

# Models to try, in order, from config.yaml (`model` + `model_fallbacks`). Groq retires
# models on a rolling basis and a retired one 404s on every call, so we never hardcode a
# single name: the run degrades to the next model instead of dying.
# Current deprecations: https://console.groq.com/docs/deprecations
MODEL_CHAIN = CFG.model_chain


def _build_system_prompt() -> str:
    """根据 config.yaml 组装研究 Agent 的系统提示词（中文）。"""
    topics = "\n".join(f"- {t}" for t in CFG.topics)
    prompt = f"""你是一个自主的 AI 新闻研究 Agent。今天的任务是：为用户找到当天最重要、
最新的 {TARGET_ARTICLES} 条 AI / Agent 领域新闻，然后交给写作环节。

工作方式：
- 调用 `search_news` 搜索网络。围绕用户的兴趣多搜几次：
{topics}
读搜索结果、判断质量，需要时再追搜。
- 只选内容扎实的单篇新闻。拒绝索引页、新闻合集、首页和标题含糊的页面
（比如 "New Ship News"）——它们不是真正的新闻。
- 每条结果都带有 `already_sent: true` 标记，表示它已在往期简报里发过。
绝不要选已经发过的文章。"""

    if FOCUS:
        prompt += f"""
- 强制要求：{TARGET_ARTICLES} 条新闻里必须恰好有一条是关于 \
{FOCUS['label']} 的。"""
        if FOCUS["priority_query"]:
            prompt += f""" 先专门搜索 "{FOCUS['priority_query']}" —— \
如果有相关的新鲜新闻就优先选它；没有的话，再选另一条有分量的 \
{FOCUS['label']} 新闻。务必跑一次 "{FOCUS['priority_query']}" 搜索和一次 \
通用的 {FOCUS['label']} 搜索。"""

    closing_focus = f"（其中一条关于 {FOCUS['label']}）" if FOCUS else ""
    prompt += f"""
- 当你找到 {TARGET_ARTICLES} 条最强、互不重复的新闻{closing_focus}后，调用 \
`finalize_selection`，把它们的准确 URL 原样从搜索结果里抄过来。

你只负责研究和挑选，不写摘要——摘要在后面的环节写。"""
    return prompt


SYSTEM_PROMPT = _build_system_prompt()


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": (
                "按关键词搜索最近 7 天的新闻。返回文章列表，含标题、链接、"
                "摘要、日期和是否已发过的标记。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "新闻搜索关键词，例如 'Claude Gemini 新模型发布'。",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_selection",
            "description": "找到有分量、互不重复的新闻后，提交选中的 3 篇文章。",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "恰好 3 个 URL，原样从搜索结果里抄过来。",
                    }
                },
                "required": ["urls"],
            },
        },
    },
]

# 写作格式约定：每条摘要固定两段，中文写成。只改这里，提示词和渲染
# 都会跟着变；测试会检查这两行标记一直在。
SYNTHESIS_FORMAT_GUIDE = (
    "每条摘要固定写两段，都用简体中文：\n"
    "第一段以“发生了什么：”开头，一句话讲清这条新闻发生了什么；\n"
    "第二段以“为什么重要（产品视角）：”开头，一到两句，从产品经理的视角\n"
    "讲这条新闻为什么值得注意（对做 AI / Agent 产品有什么启发）。\n"
    "只写片段里有的事实，不编造、不推断片段之外的事；片段太薄就只写能\n"
    "确认的部分。语气专业、中性，不煽情、不给建议。"
)

SYNTHESIS_SYSTEM = (
    "你是一名科技记者兼 AI 产品观察者。你写的简报摘要只基于每篇文章给定"
    "的来源片段，用简体中文写。专业、中性、不煽情，不写空话，不给建议，"
    "不针对任何读者。"
)


# --- API clients -----------------------------------------------------------

def _tavily_client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set")
    return TavilyClient(api_key=api_key)


def _groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return Groq(api_key=api_key)


# --- Tool implementation ---------------------------------------------------

# Tavily free-tier is touchy about bursts: firing several searches in the same
# second can trip its rate limit (which surfaces as an empty-message error). We keep
# a minimum spacing between calls and retry transient failures with backoff, because a
# single failed search can sink the whole briefing — with no real results the LLM has
# no URLs to pick and hallucinates them, so everything gets dropped as "fabricated".
_MIN_SEARCH_INTERVAL = 1.2   # seconds between consecutive Tavily calls
_SEARCH_ATTEMPTS = 3         # attempts per search before giving up
_last_search_ts = 0.0


def _throttle_search() -> None:
    """Space Tavily calls at least _MIN_SEARCH_INTERVAL apart (free-tier friendly)."""
    global _last_search_ts
    wait = _MIN_SEARCH_INTERVAL - (time.time() - _last_search_ts)
    if wait > 0:
        time.sleep(wait)
    _last_search_ts = time.time()


def search_news(query: str, max_results: int = 5) -> list[dict]:
    """Tavily search returning compact, dedup-annotated results for the agent.

    Throttled and retried with backoff: a transient Tavily blip or a free-tier
    rate-limit must not silently zero out a search (and through it the whole day).
    """
    response = None
    for attempt in range(1, _SEARCH_ATTEMPTS + 1):
        _throttle_search()
        try:
            client = _tavily_client()
            response = client.search(
                query=query,
                search_depth="advanced",
                topic="news",
                days=7,
                max_results=max_results,
            )
            break
        except Exception as exc:  # noqa: BLE001
            # str(exc) is often empty for Tavily rate-limit errors, so log the type too.
            logger.error(
                "Tavily search failed for '%s' (attempt %d/%d): %s: %s",
                query, attempt, _SEARCH_ATTEMPTS, type(exc).__name__, exc or "<no message>",
            )
            if attempt < _SEARCH_ATTEMPTS:
                time.sleep(2 * attempt)  # 2s then 4s — clears brief blips / rate limits
    if response is None:
        return []

    results = []
    for item in response.get("results", []):
        url = item.get("url", "")
        title = item.get("title", "Untitled")
        results.append(
            {
                "title": title,
                "url": url,
                "snippet": (item.get("content", "") or "")[:500],
                "date": item.get("published_date", ""),
                "already_sent": history.is_seen(url, title),
            }
        )
    return results


# --- Adaptive pacing (respect Groq free-tier TPM) --------------------------

def _pace(total_tokens: int) -> None:
    """Sleep just enough so token usage stays under the per-minute ceiling."""
    budget = TPM_LIMIT * TPM_SAFETY
    sleep_s = min(55.0, (total_tokens / budget) * 60.0)
    if sleep_s > 0:
        logger.info("Pacing: sleeping %.1fs (last call used %d tokens)", sleep_s, total_tokens)
        time.sleep(sleep_s)


# --- Helpers ---------------------------------------------------------------

def _norm_url(url: str) -> str:
    try:
        p = urlparse(url or "")
        host = p.netloc.lower()
        host = host[4:] if host.startswith("www.") else host
        return f"{host}{p.path.rstrip('/').lower()}"
    except Exception:  # noqa: BLE001
        return (url or "").strip().lower()


_UA = {"User-Agent": "Mozilla/5.0 (compatible; daily-tech-briefing/1.0)"}


def _url_alive(url: str, timeout: int = 6) -> bool:
    """Best-effort liveness check. Only a clear 404/410 counts as dead.

    Transient errors and bot-blocking statuses (403/405/429) are treated as alive
    so we never drop a good, fresh article over a flaky check.
    """
    if not url:
        return False
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(url, method=method, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return getattr(resp, "status", 200) not in (404, 410)
        except urllib.error.HTTPError as exc:
            return exc.code not in (404, 410)
        except Exception:  # noqa: BLE001 - HEAD refused or transient; try GET, then assume alive
            continue
    return True


# --- Model resolution & graceful degradation -------------------------------

# Index into MODEL_CHAIN of the model currently believed to work. Once a model is
# found to be gone we stop retrying it for the rest of the run.
_active_model = 0

# What a retired Groq model looks like: a 404 whose body carries one of these codes.
# See https://console.groq.com/docs/deprecations for what is being retired next.
_MODEL_GONE_MARKERS = (
    "model_not_found",
    "model_decommissioned",
    "does not exist or you do not have access",
    "has been decommissioned",
)


def _is_model_unavailable(exc: Exception) -> bool:
    """True if this error means "that model is gone", not "that call failed"."""
    text = str(exc).lower()
    if any(marker in text for marker in _MODEL_GONE_MARKERS):
        return True
    return getattr(exc, "status_code", None) == 404


def _chat_completion(client, **kwargs):
    """Run a Groq completion, degrading through MODEL_CHAIN if a model is retired.

    Deprecations are the one failure mode that takes the agent down permanently and
    silently (every call 404s from the day the model is decommissioned), so instead of
    failing we log loudly and try the next model. Only when the whole chain is gone
    does the run fail — and then the error names every model tried.
    """
    global _active_model

    errors: list[str] = []
    for index in range(_active_model, len(MODEL_CHAIN)):
        model = MODEL_CHAIN[index]
        try:
            response = client.chat.completions.create(model=model, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if not _is_model_unavailable(exc):
                raise
            errors.append(f"{model}: {exc}")
            logger.warning(
                "Groq model %r is unavailable (deprecated/decommissioned?): %s. "
                "Falling back to the next model in config.yaml. "
                "Check https://console.groq.com/docs/deprecations",
                model, exc,
            )
            continue

        if index != _active_model:
            logger.warning("Now using fallback model %r for the rest of this run", model)
            _active_model = index
        return response

    raise RuntimeError(
        "Every Groq model in the fallback chain failed as unavailable "
        f"({', '.join(MODEL_CHAIN)}). Update `model` / `model_fallbacks` in config.yaml "
        "with a live model from https://console.groq.com/docs/deprecations. "
        "Details: " + " | ".join(errors)
    )


def _create_completion(client, messages: list, tool_choice, max_retries: int = 4):
    """Call Groq tool-use, retrying the flaky 'tool_use_failed' parse error.

    The model occasionally emits a malformed function-call string that Groq rejects
    with a 400 tool_use_failed. A simple retry almost always recovers.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return _chat_completion(
                client,
                messages=messages,
                tools=TOOLS,
                tool_choice=tool_choice,
                temperature=0.5,
                max_tokens=1024,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if "tool_use_failed" in str(exc) or "Failed to call a function" in str(exc):
                logger.warning("Groq tool-call parse failed (attempt %d/%d), retrying", attempt + 1, max_retries)
                time.sleep(2)
                continue
            raise
    raise last_exc


def _assistant_message_to_dict(msg) -> dict:
    out = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return out


def _resolve_selection(urls: list[str], collected: list[dict]) -> list[dict]:
    """Map the agent's chosen URLs to real articles, validate, and backfill to 3.

    Only URLs that actually appeared in search results are kept (no fabricated
    links). If fewer than TARGET_ARTICLES survive, fill from the best remaining
    real, non-duplicate articles so the briefing always has 3 stories.
    """
    by_url = {_norm_url(a["url"]): a for a in collected if a.get("url")}

    chosen: list[dict] = []
    used: set[str] = set()
    for url in urls:
        key = _norm_url(url)
        article = by_url.get(key)
        if not article:
            logger.warning("Dropped fabricated/unknown URL from selection: %s", url)
            continue
        if key in used or article.get("already_sent"):
            continue
        if not _url_alive(article.get("url", "")):
            logger.warning("Dropped dead link from selection: %s", article.get("url", ""))
            continue
        chosen.append(article)
        used.add(key)

    if len(chosen) < TARGET_ARTICLES:
        for article in collected:
            key = _norm_url(article.get("url", ""))
            if key in used or article.get("already_sent"):
                continue
            if len(article.get("title", "")) < 15:  # skip nav/index-like entries
                continue
            if not _url_alive(article.get("url", "")):
                continue
            chosen.append(article)
            used.add(key)
            if len(chosen) >= TARGET_ARTICLES:
                break

    return chosen[:TARGET_ARTICLES]


def _focus_score(article: dict) -> int:
    """Rank an article against the configured focus theme.

    2 if it matches the focus priority subject, 1 if it matches any focus keyword,
    else 0. Always 0 when no focus theme is configured.
    """
    if not FOCUS:
        return 0
    text = f"{article.get('title', '')} {article.get('snippet', '')}".lower()
    pq = FOCUS["priority_query"].lower()
    if pq and pq in text:
        return 2
    return 1 if any(kw in text for kw in FOCUS_KEYWORDS) else 0



def _ensure_focus_news(selected: list[dict], collected: list[dict]) -> list[dict]:
    """Guarantee MIN_FOCUS_ARTICLES stories about the configured focus theme (prefer its priority subject).

    No-op when no focus is configured. If the agent's pick already contains enough focus
    stories, leave it. Otherwise find the best focus candidates from the gathered results,
    or run a dedicated search, and substitute them for the least-prioritised articles.
    """
    if not FOCUS or MIN_FOCUS_ARTICLES == 0:
        return selected

    # Count how many focus articles are already in the selection
    focus_count = sum(1 for a in selected if _focus_score(a) >= 1)
    needed = MIN_FOCUS_ARTICLES - focus_count

    if needed <= 0:
        return selected

    used = {_norm_url(a.get("url", "")) for a in selected}

    def _eligible(a: dict) -> bool:
        return (
            _norm_url(a.get("url", "")) not in used
            and not a.get("already_sent")
            and len(a.get("title", "")) >= 15
            and _focus_score(a) >= 1
            and _url_alive(a.get("url", ""))
        )

    candidates = [a for a in collected if _eligible(a)]
    if not candidates:
        queries = []
        if FOCUS["priority_query"]:
            queries.append(f"{FOCUS['priority_query']} news")
        queries.append(f"{FOCUS['label']} news")
        for query in queries:
            for a in search_news(query):
                if _eligible(a):
                    candidates.append(a)
            if candidates:
                break

    if not candidates:
        logger.warning("Could not find enough focus stories (%s) to guarantee coverage", FOCUS["label"])
        return selected

    # Sort candidates by focus score (priority subject first) and then by recency/quality
    candidates.sort(key=_focus_score, reverse=True)

    # Replace the least-prioritised articles in selected with the best focus candidates
    non_focus_indices = [i for i, a in enumerate(selected) if _focus_score(a) == 0]
    for i, idx in enumerate(non_focus_indices[:needed]):
        if i < len(candidates):
            best = candidates[i]
            logger.info("Injecting focus story for guaranteed coverage: %s", best.get("title", ""))
            selected[idx] = best

    return selected


    def _eligible(a: dict) -> bool:
        return (
            _norm_url(a.get("url", "")) not in used
            and not a.get("already_sent")
            and len(a.get("title", "")) >= 15
            and _focus_score(a) >= 1
            and _url_alive(a.get("url", ""))
        )

    candidates = [a for a in collected if _eligible(a)]
    if not candidates:
        queries = []
        if FOCUS["priority_query"]:
            queries.append(f"{FOCUS['priority_query']} news")
        queries.append(f"{FOCUS['label']} news")
        for query in queries:
            for a in search_news(query):
                if _eligible(a):
                    candidates.append(a)
            if candidates:
                break

    if not candidates:
        logger.warning("Could not find a focus story (%s) to guarantee coverage", FOCUS["label"])
        return selected

    candidates.sort(key=_focus_score, reverse=True)  # priority subject (2) before generic (1)
    best = candidates[0]
    logger.info("Injecting focus story for guaranteed coverage: %s", best.get("title", ""))
    selected[-1] = best
    return selected


# --- The research agent loop -----------------------------------------------

def run_agent_selection(today: str) -> tuple[list[dict], list[dict]]:
    """Run the tool-using research loop.

    Returns (selected, collected): the chosen real articles and every article seen
    across all searches (used to guarantee shipping coverage afterwards).
    """
    client = _groq_client()

    avoid = history.recent_titles(days=14)
    avoid_block = ""
    if avoid:
        listed = "\n".join(f"- {t}" for t in avoid[-40:])
        avoid_block = (
            "\n\nThese stories were already covered in recent briefings. Do NOT pick "
            "them again, and also avoid any article that tells essentially the SAME "
            "story (same event, even from a different outlet or with a different "
            f"headline):\n{listed}"
        )

    focus_clause = ""
    if FOCUS:
        focus_clause = f", one of which must be about {FOCUS['label']}"
        if FOCUS["priority_query"]:
            focus_clause += f" (search {FOCUS['priority_query']} first)"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Today is {today}. Find the {TARGET_ARTICLES} most important recent "
                f"tech stories{focus_clause}. Start searching now.{avoid_block}"
            ),
        },
    ]

    searches_used = 0
    collected: list[dict] = []

    for step in range(MAX_STEPS):
        force_final = step == MAX_STEPS - 1 or searches_used >= MAX_SEARCHES
        tool_choice = (
            {"type": "function", "function": {"name": "finalize_selection"}}
            if force_final
            else "auto"
        )

        try:
            response = _create_completion(client, messages, tool_choice)
        except Exception as exc:  # noqa: BLE001
            if not force_final:
                raise
            # On the forced-finalize step the model sometimes tries to search anyway,
            # and Groq rejects the whole request rather than the stray tool call. Every
            # article it found is already in `collected`, so finish from those instead
            # of losing the briefing at the last step.
            logger.warning(
                "Forced finalize failed (%s). Selecting from the %d article(s) already "
                "gathered instead.", exc, len(collected),
            )
            return _resolve_selection([], collected), collected

        msg = response.choices[0].message
        messages.append(_assistant_message_to_dict(msg))

        if not msg.tool_calls:
            messages.append(
                {"role": "user", "content": "Continue: search_news again or call finalize_selection."}
            )
            _pace(getattr(response.usage, "total_tokens", 3000))
            continue

        for tc in msg.tool_calls:
            if tc.function.name == "finalize_selection":
                try:
                    urls = json.loads(tc.function.arguments).get("urls", [])
                except Exception:  # noqa: BLE001
                    urls = []
                logger.info("Agent finalized selection after %d search(es)", searches_used)
                return _resolve_selection(urls, collected), collected

        for tc in msg.tool_calls:
            if tc.function.name != "search_news":
                continue
            try:
                query = json.loads(tc.function.arguments).get("query", "")
            except Exception:  # noqa: BLE001
                query = ""

            if searches_used >= MAX_SEARCHES:
                content = "Search budget exhausted. Call finalize_selection now."
            else:
                searches_used += 1
                results = search_news(query)
                collected.extend(results)
                logger.info("Search %d/%d: '%s' -> %d results", searches_used, MAX_SEARCHES, query, len(results))
                content = json.dumps(results, ensure_ascii=False)

            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "name": "search_news", "content": content}
            )

        _pace(getattr(response.usage, "total_tokens", 3000))

    raise RuntimeError("Agent finished without finalizing a selection")


# --- Phase 2: write the briefing from the selected real articles -----------

_RECOVERABLE_SYNTHESIS_MARKERS = (
    "json_validate_failed",     # output was not valid JSON (usually truncated by the budget)
    "reasoning_effort",         # a model in the chain rejects the parameter
)


def _is_recoverable_synthesis_error(exc: Exception) -> bool:
    """True for errors a different call shape can fix; model outages are not these."""
    text = str(exc).lower()
    return any(m in text for m in _RECOVERABLE_SYNTHESIS_MARKERS)


def _parse_json_object(text: str) -> dict:
    """Parse a JSON object out of model output, tolerating leaked reasoning.

    Some models wrap their answer in <think> blocks or prose even when asked for JSON
    only. Strip that and take the outermost {...}; raise if there is none.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model output")
    data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("model output is not a JSON object")
    return data


def _synthesis_completion(client, messages: list) -> dict | None:
    """Get the writing call's JSON, degrading the call shape rather than the briefing.

    Attempt 1 is the cheap, strict one. If the API refuses the JSON (budget exhausted,
    parameter rejected), later attempts loosen the request. None means every attempt
    failed and the caller should fall back to snippet summaries: a briefing with plain
    excerpts is still worth sending, a failure email is not.
    """
    attempts = (
        # Strict JSON mode, low reasoning effort: halves the budget on gpt-oss models.
        {"response_format": {"type": "json_object"}, "reasoning_effort": "low"},
        # Same without the effort hint, in case a model in the chain rejects it.
        {"response_format": {"type": "json_object"}},
        # No JSON mode at all; parse whatever comes back leniently.
        {},
    )
    for index, extra in enumerate(attempts, start=1):
        try:
            response = _chat_completion(
                client, messages=messages, temperature=0.5,
                max_tokens=SYNTHESIS_MAX_TOKENS, **extra,
            )
            return _parse_json_object(response.choices[0].message.content)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Synthesis attempt %d returned unparseable JSON: %s", index, exc)
        except Exception as exc:  # noqa: BLE001
            if not _is_recoverable_synthesis_error(exc):
                raise
            logger.warning("Synthesis attempt %d refused by the API: %s", index, str(exc)[:200])
    return None


def synthesize_briefing(selected: list[dict], today: str) -> dict:
    """写中文摘要（只依据片段）+ 每日一词、每日一句。"""
    numbered = "\n\n".join(
        f"[{i}] 标题：{a.get('title', '')}\n片段：{a.get('snippet', '')}"
        for i, a in enumerate(selected, start=1)
    )

    used_words = history.recent_words()
    used_quotes = history.recent_quotes()
    avoid_words = f" 不要重复最近用过的这些词：{', '.join(used_words)}。" if used_words else ""
    avoid_quotes = (
        f" 不要重复最近用过的这些名言：{' | '.join(used_quotes)}。" if used_quotes else ""
    )

    user = f"""今天是 {today}。下面是选中的 {len(selected)} 篇文章：

{numbered}

{SYNTHESIS_FORMAT_GUIDE}

摘要只围绕标题那一个新闻写：忽略片段里出现的无关标题、“最近新闻还包括……”、
合集类内容。不要编造片段之外的事实；片段太薄就只总结标题新闻里能确认的
部分。不要写空话、不要发表观点、不要写“为什么重要”之外的发挥，换着花样写。

然后再加：
- "word_of_the_day"：一个真实、具体的技术术语（不要像 "AI" 这么泛；比如
  推理、嵌入、幂等、向量数据库、微调、Webhook、延迟、量化……），配一句中文
  定义和一句自然的中文例句。{avoid_words}
- "quote_of_the_day"：一句真实、可查的科技界名人的话，配作者名字。{avoid_quotes}

只返回 JSON，格式如下：
{{
  "summaries": {{ "1": "发生了什么：……为什么重要（产品视角）：……", "2": "...", "3": "..." }},
  "word_of_the_day": {{ "word": "...", "definition": "...", "example": "..." }},
  "quote_of_the_day": {{ "text": "...", "author": "..." }}
}}"""

    client = _groq_client()
    data = _synthesis_completion(
        client,
        [
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    if data is None:
        # Every call shape failed. Ship the real articles with their source excerpts
        # rather than nothing: the reader still gets the stories and the links.
        logger.warning("Synthesis failed on every attempt; sending snippet summaries without word/quote")
        data = {}

    summaries = data.get("summaries", {})
    news = []
    for i, article in enumerate(selected, start=1):
        news.append(
            {
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "date": article.get("date", ""),
                "summary": summaries.get(str(i), "") or article.get("snippet", "")[:300],
            }
        )

    return {
        "sections": [{"title": "今日 AI / Agent 要闻", "news": news}],
        "word_of_the_day": data.get("word_of_the_day", {}),
        "quote_of_the_day": data.get("quote_of_the_day", {}),
    }


# --- HTML rendering --------------------------------------------------------

def _domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:  # noqa: BLE001
        return ""


def _format_date(raw: str) -> str:
    """Format a publication date as 'May 28, 2026'; fall back to the raw value."""
    if not raw:
        return ""
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%a, %d %b %Y %H:%M:%S %Z"):
        try:
            return datetime.strptime(raw, fmt).strftime("%B %d, %Y")
        except ValueError:
            continue
    return raw[:24]


# --- 信源可信度徽章 ---------------------------------------------------------
# 只看链接域名，不调任何接口，离线可测。分三档：
#   官方一手：厂商 / 项目自己的博客、文档、GitHub 仓库页，消息的源头；
#   主流媒体：有采编团队的主流科技 / 财经媒体；
#   待核实：其他一切来源，读的时候多留个心眼。
# 域名表写在这里，方便以后自己增减。匹配按“等于或以 .域名 结尾”算，
# 所以 www.reuters.com、blog.google 都能命中。
_OFFICIAL_DOMAINS = (
    "openai.com",
    "anthropic.com",
    "blog.google",
    "deepmind.google",
    "developers.googleblog.com",
    "research.google",
    "github.blog",
    "github.com",
    "docs.github.com",
    "microsoft.com",
    "blogs.microsoft.com",
    "devblogs.microsoft.com",
    "aws.amazon.com",
    "amazon.science",
    "ai.meta.com",
    "meta.com",
    "nvidia.com",
    "huggingface.co",
    "langchain.com",
    "tavily.com",
    "groq.com",
    "cloudflare.com",
    "vercel.com",
    "supabase.com",
)

_MEDIA_DOMAINS = (
    "reuters.com",
    "bloomberg.com",
    "ft.com",
    "wsj.com",
    "theverge.com",
    "techcrunch.com",
    "wired.com",
    "arstechnica.com",
    "theinformation.com",
    "engadget.com",
    "venturebeat.com",
    "zdnet.com",
    "sina.com.cn",
    "36kr.com",
    "jiemian.com",
    "huxiu.com",
    "geekpark.net",
    "infoq.cn",
    "pingwest.com",
)

# 徽章颜色：官方一手用深蓝，主流媒体用灰蓝，待核实用浅黄底深字。
_BADGE_STYLES = {
    "official": "background-color:#0B2C4D;color:#ffffff;",
    "media": "background-color:#5B7A99;color:#ffffff;",
    "unverified": "background-color:#FFF3CD;color:#7A5B00;",
}


def source_credibility(url: str) -> tuple[str, str]:
    """给一条新闻链接定信源档位，返回 (徽章文字, 档位英文名)。

    档位只有三档：official（官方一手）、media（主流媒体）、
    unverified（待核实）。链接为空或解析不出域名时，一律算待核实。
    """
    domain = _domain(url).lower()
    if not domain:
        return "待核实", "unverified"
    for base in _OFFICIAL_DOMAINS:
        if domain == base or domain.endswith("." + base):
            return "官方一手", "official"
    for base in _MEDIA_DOMAINS:
        if domain == base or domain.endswith("." + base):
            return "主流媒体", "media"
    return "待核实", "unverified"


def _badge_html(url: str) -> str:
    """把徽章画成 HTML 里的小标签；返回空字符串表示不画（不会发生）。"""
    label, level = source_credibility(url)
    style = _BADGE_STYLES[level]
    return (
        f'<span style="display:inline-block;font-family:Arial,Helvetica,sans-serif;'
        f"font-size:11px;font-weight:bold;letter-spacing:1px;padding:2px 8px;"
        f"border-radius:10px;{style}\">{label}</span>"
    )


def _split_summary(summary: str) -> tuple[str, str]:
    """把摘要拆成（发生了什么， 为什么重要）两段。

    模型按固定格式写，中间以“为什么重要（产品视角）："分隔；拆不开时
    （比如写作失败回退到原文片段），整段都算“发生了什么”。
    """
    marker = "为什么重要（产品视角）："
    if marker in summary:
        what, why = summary.split(marker, 1)
        what = what.replace("发生了什么：", "", 1).strip()
        return what, why.strip()
    return summary.strip(), ""


def _news_item_html(item: dict) -> str:
    title = item.get("title", "Untitled")
    url = item.get("url", "#")
    summary = item.get("summary", "")
    source = _domain(url)
    pub_date = _format_date(item.get("date", ""))
    meta = f"来源：{source}"
    if pub_date:
        meta += f" &nbsp;&middot;&nbsp; {pub_date}"
    badge = _badge_html(url)
    what, why = _split_summary(summary)
    summary_html = (
        f'<p style="margin:0 0 8px;font-family:Georgia,\'Times New Roman\',serif;font-size:15px;'
        f"line-height:1.65;color:#2b2b2b;text-align:justify;\"><b>发生了什么：</b>{what}</p>"
    )
    if why:
        summary_html += (
            f'<p style="margin:0 0 8px;font-family:Georgia,\'Times New Roman\',serif;font-size:15px;'
            f"line-height:1.65;color:#2b2b2b;text-align:justify;\"><b>为什么重要（产品视角）：</b>{why}</p>"
        )
    return f"""
        <div style="margin:0 0 26px;">
          <a href="{url}" style="color:#111111;text-decoration:none;">
            <h3 style="margin:0 0 8px;font-family:Georgia,'Times New Roman',serif;
                       font-size:20px;font-weight:700;line-height:1.3;color:#111111;">{title}</h3>
          </a>
          {badge}
          {summary_html}
          <a href="{url}" style="font-family:Arial,Helvetica,sans-serif;font-size:11px;
                    letter-spacing:1px;color:#0B2C4D;
                    text-decoration:none;font-weight:bold;">{meta} &rsaquo;</a>
        </div>"""


def _section_html(section: dict) -> str:
    title = section.get("title", "")
    items_html = "".join(_news_item_html(n) for n in section.get("news", []))
    return f"""
        <div style="margin-top:34px;">
          <div style="border-top:2px solid #111111;border-bottom:1px solid #111111;
                      padding:6px 0;margin-bottom:20px;">
            <span style="font-family:Arial,Helvetica,sans-serif;font-size:13px;font-weight:bold;
                         letter-spacing:2px;text-transform:uppercase;color:#111111;">{title}</span>
          </div>
          {items_html}
        </div>"""


def _render_html(briefing: dict, today: str) -> str:
    """Render the briefing dict into a newspaper-style HTML email."""
    sections_html = "".join(_section_html(s) for s in briefing.get("sections", []))

    # 底部不写个人信息——这行字会出现在每个 clone 出来的简报里。
    footer = "由你的 AI Agent 精选"
    if CFG.footer_note:
        footer += f" &nbsp;|&nbsp; {CFG.footer_note}"

    word = briefing.get("word_of_the_day", {})
    quote = briefing.get("quote_of_the_day", {})

    # Either block is omitted entirely when synthesis could not produce it, rather
    # than rendering an empty box.
    word_html = f"""
        <div style="margin-top:34px;border:1px solid #c9c2b4;background-color:#f6f3ec;padding:20px;">
          <div style="font-family:Arial,Helvetica,sans-serif;font-size:11px;font-weight:bold;
                      letter-spacing:2px;text-transform:uppercase;color:#0B2C4D;margin-bottom:10px;">
            每日一词
          </div>
          <div style="font-family:Georgia,'Times New Roman',serif;font-size:22px;font-weight:700;
                      color:#111111;margin-bottom:6px;">{word.get('word', '')}</div>
          <p style="margin:0 0 8px;font-family:Georgia,'Times New Roman',serif;font-size:15px;
                    line-height:1.6;color:#2b2b2b;">{word.get('definition', '')}</p>
          <p style="margin:0;font-family:Georgia,'Times New Roman',serif;font-size:14px;
                    line-height:1.6;color:#555555;font-style:italic;">&ldquo;{word.get('example', '')}&rdquo;</p>
        </div>"""

    quote_html = f"""
        <div style="margin-top:28px;border-top:1px solid #ddd;border-bottom:1px solid #ddd;
                    padding:24px 16px;text-align:center;">
          <p style="margin:0 0 10px;font-family:Georgia,'Times New Roman',serif;font-size:19px;
                    line-height:1.5;color:#111111;font-style:italic;">&ldquo;{quote.get('text', '')}&rdquo;</p>
          <p style="margin:0;font-family:Arial,Helvetica,sans-serif;font-size:12px;
                    letter-spacing:1px;text-transform:uppercase;color:#777777;">— {quote.get('author', '')}</p>
        </div>"""

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background-color:#e9e6df;">
  <div style="max-width:640px;margin:0 auto;background-color:#ffffff;
              border-left:1px solid #ddd;border-right:1px solid #ddd;">
    <div style="padding:30px 36px 18px;text-align:center;border-bottom:4px double #111111;">
      <div style="font-family:Arial,Helvetica,sans-serif;font-size:11px;letter-spacing:3px;
                  text-transform:uppercase;color:#0B2C4D;margin-bottom:8px;">由你的 AI Agent 精选</div>
      <h1 style="margin:0;font-family:Georgia,'Times New Roman',serif;font-size:38px;
                 font-weight:700;letter-spacing:1px;color:#111111;line-height:1.1;">{CFG.title}</h1>
      <div style="font-family:Arial,Helvetica,sans-serif;font-size:12px;letter-spacing:2px;
                  text-transform:uppercase;color:#555555;margin-top:10px;">{today}</div>
    </div>
    <div style="padding:6px 36px 30px;">
      {sections_html}
      {word_html if word.get("word") else ""}
      {quote_html if quote.get("text") else ""}
    </div>
    <div style="padding:20px 36px;border-top:2px solid #111111;text-align:center;">
      <p style="margin:0;font-family:Arial,Helvetica,sans-serif;font-size:11px;letter-spacing:1px;
                color:#888888;">{footer}</p>
    </div>
  </div>
</body>
</html>"""


def render_text(subject: str, payload: dict) -> str:
    """把简报渲染成纯文本，给 --dry-run 在终端预览用。"""
    lines = [subject, "=" * len(subject), ""]
    for i, article in enumerate(payload.get("articles", []), start=1):
        badge = article.get("badge", "")
        badge_text = f"【{badge}】" if badge else ""
        lines += [
            f"{i}. {article.get('title', '')} {badge_text}",
            f"   {article.get('summary', '')}",
            f"   -> {article.get('url', '')}",
            "",
        ]

    word = payload.get("word", {})
    if word.get("word"):
        lines += [
            f"每日一词 — {word.get('word', '')}",
            f"   {word.get('definition', '')}",
            f"   \"{word.get('example', '')}\"",
            "",
        ]

    quote = payload.get("quote", {})
    if quote.get("text"):
        lines += [f"\"{quote.get('text', '')}\"", f"   — {quote.get('author', '')}", ""]

    return "\n".join(lines)


# --- Public pipeline -------------------------------------------------------

def build_briefing() -> tuple[str, str, dict]:
    """Run the agent, write the briefing, render the email.

    Returns (subject, html, payload). `payload` holds everything to persist after a
    successful send: {"articles": [...], "word": {...}, "quote": {...}}.
    """
    today = datetime.now().strftime("%B %d, %Y")
    subject = f"{CFG.title} — {today}"
    logger.info("Model chain: %s", " -> ".join(MODEL_CHAIN))

    selected, collected = run_agent_selection(today)
    selected = _ensure_focus_news(selected, collected)
    if not selected:
        raise RuntimeError("Agent selected no valid articles")

    briefing = synthesize_briefing(selected, today)
    html_body = _render_html(briefing, today)
    written = [n for section in briefing.get("sections", []) for n in section.get("news", [])]
    payload = {
        # `summary` 跟着走，这样 --dry-run 不用再调一次模型就能打印纯文本预览；
        # history.record_sent 只读 url/title。`badge` 是信源可信度徽章文字。
        "articles": [
            {
                "url": n.get("url", ""),
                "title": n.get("title", ""),
                "summary": n.get("summary", ""),
                "badge": source_credibility(n.get("url", ""))[0],
            }
            for n in written
        ],
        "word": briefing.get("word_of_the_day", {}),
        "quote": briefing.get("quote_of_the_day", {}),
    }
    logger.info("Briefing built: %d article(s), word=%r", len(payload["articles"]), payload["word"].get("word", ""))
    return subject, html_body, payload
