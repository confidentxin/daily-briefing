"""信源可信度徽章：只看域名分档，不调接口，离线可测。

三档：官方一手 / 主流媒体 / 待核实。域名表在 agent.py 里，改表不用改逻辑。
"""

import agent


def test_official_vendor_blogs_are_first_party():
    assert agent.source_credibility("https://openai.com/index/introducing-gpt-5") == (
        "官方一手",
        "official",
    )
    assert agent.source_credibility("https://www.anthropic.com/news/claude-4") == (
        "官方一手",
        "official",
    )
    assert agent.source_credibility("https://blog.google/technology/ai/") == (
        "官方一手",
        "official",
    )


def test_github_repo_pages_count_as_first_party():
    # 仓库页就是项目本身，是最一手的信源。
    assert agent.source_credibility("https://github.com/langchain-ai/langgraph") == (
        "官方一手",
        "official",
    )


def test_subdomains_match_their_parent_domain():
    assert agent.source_credibility("https://www.reuters.com/technology/") == (
        "主流媒体",
        "media",
    )
    assert agent.source_credibility("https://www.36kr.com/p/123456") == (
        "主流媒体",
        "media",
    )


def test_unknown_sources_are_flagged_unverified():
    assert agent.source_credibility("https://some-random-blog.example.com/post") == (
        "待核实",
        "unverified",
    )
    assert agent.source_credibility("") == ("待核实", "unverified")
    assert agent.source_credibility("not a url") == ("待核实", "unverified")


def test_format_guide_pins_the_two_part_summary():
    # 写作格式约定是中文版的核心：两段式，标记不能丢。
    assert "发生了什么：" in agent.SYNTHESIS_FORMAT_GUIDE
    assert "为什么重要（产品视角）：" in agent.SYNTHESIS_FORMAT_GUIDE


def test_summary_splits_on_the_importance_marker():
    what, why = agent._split_summary(
        "发生了什么：A 发布了 B。为什么重要（产品视角）：这会改变 C。"
    )
    assert what == "A 发布了 B。"
    assert why == "这会改变 C。"


def test_plain_snippet_fallback_stays_whole():
    # 写作失败回退到原文片段时，整段都算“发生了什么”，不硬拆。
    what, why = agent._split_summary("一段没有格式标记的原文片段。")
    assert what == "一段没有格式标记的原文片段。"
    assert why == ""


def test_news_item_html_carries_the_badge():
    html = agent._news_item_html(
        {
            "title": "T",
            "url": "https://openai.com/index/x",
            "summary": "发生了什么：A。为什么重要（产品视角）：B。",
            "date": "2026-10-30",
        }
    )
    assert "官方一手" in html
    assert "发生了什么：" in html
    assert "为什么重要（产品视角）：" in html
