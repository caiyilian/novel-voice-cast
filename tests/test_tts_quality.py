from backend.app.core.tts_quality import (
    compact_performance_control,
    control_variants,
    duration_quality_bounds,
    duration_quality_problems,
    semantic_char_count,
    split_tts_text,
)


def test_semantic_split_preserves_source_and_limits_each_chunk():
    text = (
        "商人沿着结冰的道路继续前行，远处的钟声缓慢响起，"
        "而同行者仍在思考刚才那场危险的交易。"
        "等他们越过山口以后，天空终于露出了一点微光。"
    )

    chunks = split_tts_text(text, max_chars=32, min_chars=6)

    assert "".join(chunks) == text
    assert len(chunks) >= 3
    assert all(0 < semantic_char_count(chunk) <= 32 for chunk in chunks)


def test_compact_control_drops_quoted_dialogue_but_keeps_performance_axes():
    raw = (
        "‘嗯……’带思考停顿，‘真好吃’流露真实满足但保持克制，"
        "后半句使用正式事务性语气，音量正常，节奏舒缓。"
    )

    compact = compact_performance_control(
        raw, speaker="骑士", emotion="happy", max_chars=32
    )

    assert len(compact) <= 32
    assert "嗯" not in compact
    assert "真好吃" not in compact
    assert "克制" in compact
    assert "严肃" in compact
    assert "语速舒缓" in compact
    assert control_variants(compact)[-1] == ""


def test_duration_envelope_rejects_acceleration_and_spoken_control_leak():
    bounds = duration_quality_bounds(
        "这里的村民已不再需要咱了。",
        "平稳，语速舒缓",
        min_ratio=0.92,
        max_ratio=1.9,
    )

    assert duration_quality_problems(bounds["expected_duration_seconds"], bounds) == []
    assert "too fast" in duration_quality_problems(0.8, bounds)[0]
    assert "leaked control text" in duration_quality_problems(21.6, bounds)[0]
