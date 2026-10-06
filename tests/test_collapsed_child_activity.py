"""Real attachment/full renderer: child activity is not a parent notification."""
import pytest

from tests.test_child_session_status import run_component, session


@pytest.mark.parametrize("kind", ["fork", "delegated", "reference"])
@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("active", ["parent", "other"])
@pytest.mark.parametrize("own", ["idle", "unread", "approval", "clarify", "streaming"])
def test_collapsed_child_activity_preserves_own_notification(kind, expanded, active, own):
    parent = session("parent", own)
    child = session("child", "streaming", parent_session_id="parent",
                    relationship_type="child_session", raw_source="subagent",
                    session_source="fork" if kind == "fork" else "other")
    raw, refs = [parent, child], [parent, child]
    if kind == "reference":
        child.update(archived=True, _lineage_root_id="child")
        raw = [parent]
    out = run_component(raw, refs, expanded, active)
    expected = own != "streaming" and (not expanded or kind == "reference")
    assert len(out["activity"]) == int(expected), "Collapsed running child needs a separate parent activity spinner"
    if expected:
        assert out["activity"][0]["className"].split() == [
            "session-state-indicator", "session-child-activity-indicator", "is-streaming"]
    dot = out["dot"]["className"].split()
    assert ("is-streaming" in dot) == (own == "streaming")
    assert ("is-unread" in dot) == (own == "unread" and active != "parent")
    for state in ["approval", "clarify"]:
        assert (f"is-attention-{state}" in dot) == (own == state)
    assert ("unread" in out["parent"].split()) == (own == "unread" and active != "parent")
    assert ("needs-attention" in out["parent"].split()) == (own in ["approval", "clarify"])
    assert len(out["children"]) == int(expanded and kind != "reference")
    assert out["unchanged"]


def test_search_expansion_uses_child_rows_instead_of_parent_activity():
    parent = session("parent")
    child = session("child", "streaming", parent_session_id="parent", relationship_type="child_session")
    out = run_component([parent, child], [parent, child], False, "other", search="task")
    assert not out["activity"]
    assert len(out["children"]) == 1
    assert "streaming" in out["children"][0]["className"].split()


@pytest.mark.parametrize("attention", ["approval", "clarify", "generic"])
def test_other_child_attention_does_not_mask_collapsed_activity(attention):
    parent = session("parent", "unread")
    children = [session("running", "streaming", parent_session_id="parent", relationship_type="child_session"),
                session("waiting", attention, parent_session_id="parent", relationship_type="child_session")]
    out = run_component([parent, *children], [parent, *children], False, "other")
    assert len(out["activity"]) == 1
    assert f"is-attention-{attention}" in out["chip"]["children"][-1]["className"].split()
    assert "is-unread" in out["dot"]["className"].split()


@pytest.mark.parametrize("state", ["idle", "unread", "approval", "clarify", "generic"])
def test_settled_child_clears_activity_without_bubbling_notifications(state):
    parent = session("parent", _child_session_streaming=True)
    child = session("child", state, parent_session_id="parent", relationship_type="child_session")
    out = run_component([parent, child], [parent, child], False, "other")
    assert not out["activity"]
    assert not any(c.startswith("is-") for c in out["dot"]["className"].split())
    assert not any(c in out["parent"].split() for c in ["unread", "needs-attention", "streaming"])
