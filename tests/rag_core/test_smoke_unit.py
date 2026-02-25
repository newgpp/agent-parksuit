from agent_parksuite_rag_core.schemas.retrieve import RetrieveRequest


def test_retrieve_request_default_top_k() -> None:
    req = RetrieveRequest(query="test")
    assert req.top_k == 5
