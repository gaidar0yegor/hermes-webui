"""Regression tests for the review findings on PR #7903.

The first round of the streaming-snapshot collapse dropped every non-winner
snapshot from *both* source lists.  When the winner came from ``state.db`` and
the sidecar held a later durable row, the collapsed row disappeared from the
merge result entirely: master duplicated the row, the guard deleted it.  These
tests pin the reviewed contract instead: exactly one copy of the collapsed
``_row_id`` survives *in each list*, at the position where that list first saw
the row, carrying the most advanced payload.  A row that never reaches the
merge (settled) and mixed buckets stay untouched.
"""


def _partial_assistant(api_text: str, first_token_ms: int, row_id: int = 24318) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "timestamp": 1000.0,
        "finish_reason": "incomplete",
        "api_content": api_text,
        "_row_id": row_id,
        "_firstTokenMs": first_token_ms,
        "_turnTps": 1.0,
        "_db_persisted": True,
    }


def _settled_assistant(api_text: str, row_id: int = 24318) -> dict:
    return {
        "role": "assistant",
        "content": "visible reply",
        "timestamp": 1000.0,
        "finish_reason": "stop",
        "api_content": api_text,
        "_row_id": row_id,
        "_firstTokenMs": 10,
        "_turnTps": 1.0,
        "_db_persisted": True,
    }


def _user_row(content: str = "canonical prompt", row_id: int = 24317) -> dict:
    return {
        "role": "user",
        "content": content,
        "timestamp": 999.0,
        "_row_id": row_id,
        "_db_persisted": True,
    }


def _rows(merged):
    return [
        m.get("_row_id")
        for m in merged
        if isinstance(m, dict) and m.get("_row_id") is not None
    ]


def test_collapse_keeps_row_at_sidecar_position_when_state_holds_winner():
    """Reviewer probe row 1: state.db is a superset and the sidecar has a
    later row.  The collapsed row must survive exactly once, holding the most
    advanced snapshot, sitting between the user row and the later turn."""
    import api.models as models

    sidecar = [
        _user_row(),
        _partial_assistant("Reas:", 10),
        _partial_assistant("Reas: partial", 12),
        _user_row("next prompt", row_id=24319),
        _settled_assistant("done", row_id=24320),
    ]
    state = [
        _user_row(),
        _partial_assistant("Reas: partial longer", 14),
        _user_row("next prompt", row_id=24319),
        _settled_assistant("done", row_id=24320),
    ]

    merged = models.merge_session_messages_append_only(sidecar, state)
    rows = _rows(merged)

    assert rows.count(24318) == 1, f"row 24318 present {rows.count(24318)}x"
    kept = next(m for m in merged if m.get("_row_id") == 24318)
    assert kept["api_content"] == "Reas: partial longer"
    # positional contract: after the first user row, before the next turn
    assert rows.index(24317) < rows.index(24318) < rows.index(24319)


def test_collapse_survives_when_state_winner_is_last_row():
    """Reviewer probe row 2: one skeleton on each side plus a later sidecar
    row, winner in state.db.  Same contract: one row, most advanced payload."""
    import api.models as models

    sidecar = [
        _user_row(),
        _partial_assistant("Reas: partial", 12),
        _user_row("next prompt", row_id=24319),
    ]
    state = [
        _user_row(),
        _partial_assistant("Reas: partial longer", 14),
    ]

    merged = models.merge_session_messages_append_only(sidecar, state)
    rows = _rows(merged)

    assert rows.count(24318) == 1, f"row 24318 present {rows.count(24318)}x"
    kept = next(m for m in merged if m.get("_row_id") == 24318)
    assert kept["api_content"] == "Reas: partial longer"
    assert rows.index(24318) < rows.index(24319)


def test_collapse_multi_snapshot_sidecar_with_later_rows_stays_stable_over_repeats():
    """Reviewer probe row 1 replayed five times: row count and order must not
    drift across repeated merges (persist feeds next merge)."""
    import api.models as models

    merged = [
        _user_row(),
        _partial_assistant("Reas:", 10),
        _partial_assistant("Reas: partial", 12),
        _user_row("next prompt", row_id=24319),
        _settled_assistant("done", row_id=24320),
    ]
    state = [
        _user_row(),
        _partial_assistant("Reas: partial longer", 14),
        _user_row("next prompt", row_id=24319),
        _settled_assistant("done", row_id=24320),
    ]

    for _ in range(5):
        merged = models.merge_session_messages_append_only(merged, state)

    rows = _rows(merged)
    assert rows.count(24318) == 1, f"row 24318 grew to {rows.count(24318)} copies"
    assert rows == sorted(rows), "merge must not reorder durable rows"


def test_mixed_bucket_skeletons_and_settled_row_left_untouched():
    """Reviewer probe row 4 / second finding: a settled row sharing the durable
    ``_row_id`` with skeletons makes the bucket mixed.  Mixed buckets must stay
    exactly as they are (the docstring contract), so the non-skeleton member is
    never collapsed away."""
    import api.models as models

    sidecar = [
        _user_row(),
        _partial_assistant("Reas:", 10),
        _settled_assistant("settled variant", row_id=24318),
    ]
    state = [
        _user_row(),
        _partial_assistant("Reas: more", 13, row_id=24318),
    ]

    merged_sidecar, merged_state = models._collapse_streaming_row_id_snapshots(
        sidecar, state
    )

    # Nothing collapses: every original object is still present, unchanged.
    assert merged_sidecar == sidecar
    assert merged_state == state


def test_collapse_with_reasoning_winner_keeps_assistant_in_model_history():
    """#7903 review (CORE): the state.db winner often carries ``reasoning``
    mid-stream. Substituting the WHOLE winner row into the sidecar made
    ``_sanitize_messages_for_api()`` classify it as a reasoning-only display
    row and skip it, so the assistant turn vanished from the next request's
    model history (master: u, a, a, u; whole-row head: u, u). Only the provider
    payload may be carried forward."""
    import api.models as models
    from api.streaming import _sanitize_messages_for_api

    sidecar = [
        _user_row(),
        _partial_assistant("Reas:", 10),
        _partial_assistant("Reas: partial", 12),
        _user_row("next prompt", row_id=24319),
    ]
    winner = _partial_assistant("Reas: partial longer", 14)
    winner["reasoning"] = "thinking about it"
    state = [_user_row(), winner, _user_row("next prompt", row_id=24319)]

    merged = models.merge_session_messages_append_only(sidecar, state)
    assert _rows(merged).count(24318) == 1
    kept = next(m for m in merged if m.get("_row_id") == 24318)
    assert kept["api_content"] == "Reas: partial longer"
    assert "reasoning" not in kept

    api_history = _sanitize_messages_for_api(merged)
    roles = [m.get("role") for m in api_history]
    assert "assistant" in roles, roles
    assert roles.index("assistant") < len(roles) - 1  # before the next prompt


def test_conflicting_stable_ids_on_one_row_id_are_not_collapsed():
    """#7903 review (SILENT): snapshots sharing one ``_row_id`` but carrying
    conflicting stable ``id`` values are different messages to the merge's
    identity guard. Collapsing them would discard the ``turn-a`` payload;
    the bucket must be left untouched, exactly as master leaves it."""
    import api.models as models

    a1 = _partial_assistant("A first", 10)
    a1["id"] = "turn-a"
    a2 = _partial_assistant("A first longer", 12)
    a2["id"] = "turn-a"
    b1 = _partial_assistant("B other payload, longest of all", 14)
    b1["id"] = "turn-b"
    sidecar = [_user_row(), a1, a2]
    state = [_user_row(), b1]

    collapsed_sidecar, collapsed_state = models._collapse_streaming_row_id_snapshots(
        list(sidecar), list(state)
    )
    assert collapsed_sidecar == sidecar
    assert collapsed_state == state
    merged = models.merge_session_messages_append_only(sidecar, state)
    payloads = {m.get("api_content") for m in merged if m.get("_row_id") == 24318}
    assert "A first longer" in payloads or "A first" in payloads
