from __future__ import annotations

from typing import Any, Awaitable, Callable, TypedDict

from langgraph.graph import END, StateGraph
from loguru import logger

from agent_parksuite_rag_core.schemas.rag import HybridAnswerRequest, RetrieveResponseItem

RetrieveFn = Callable[[HybridAnswerRequest], Awaitable[list[RetrieveResponseItem]]]
ClassifyFn = Callable[[HybridAnswerRequest], Awaitable[str]]
ArrearsFactsFn = Callable[[HybridAnswerRequest], Awaitable[dict[str, Any]]]
FeeFactsFn = Callable[[HybridAnswerRequest], Awaitable[dict[str, Any]]]
SynthesizeFn = Callable[[str, list[RetrieveResponseItem], dict[str, Any], str], Awaitable[tuple[str, list[str], str]]]


class HybridGraphState(TypedDict, total=False):
    payload: HybridAnswerRequest
    intent: str
    retrieved_items: list[RetrieveResponseItem]
    business_facts: dict[str, Any]
    conclusion: str
    key_points: list[str]
    model: str
    trace: list[str]


async def run_hybrid_workflow(
    payload: HybridAnswerRequest,
    retrieve_fn: RetrieveFn,
    classify_fn: ClassifyFn,
    arrears_facts_fn: ArrearsFactsFn,
    fee_facts_fn: FeeFactsFn,
    synthesize_fn: SynthesizeFn,
    request_id: str = "",
) -> HybridGraphState:
    async def _node_intent_classifier(state: HybridGraphState) -> HybridGraphState:
        logger.info("hybrid[{}] node=intent_classifier start", request_id)
        intent = await classify_fn(state["payload"])
        logger.info("hybrid[{}] node=intent_classifier intent={}", request_id, intent)
        return {
            "intent": intent,
            "trace": [*state.get("trace", []), f"intent_classifier:{intent}"],
        }

    def _route_from_intent(state: HybridGraphState) -> str:
        intent = state.get("intent", "rule_explain")
        if intent == "arrears_check":
            return "arrears_check_flow"
        if intent == "fee_verify":
            return "fee_verify_flow"
        return "rule_explain_flow"

    async def _node_rule_explain_flow(state: HybridGraphState) -> HybridGraphState:
        logger.info("hybrid[{}] node=rule_explain_flow", request_id)
        return {
            "business_facts": {"intent": "rule_explain", "note": "RAG-only explanation flow"},
            "trace": [*state.get("trace", []), "rule_explain_flow"],
        }

    async def _node_arrears_check_flow(state: HybridGraphState) -> HybridGraphState:
        logger.info("hybrid[{}] node=arrears_check_flow start", request_id)
        facts = await arrears_facts_fn(state["payload"])
        logger.info(
            "hybrid[{}] node=arrears_check_flow arrears_count={}",
            request_id,
            facts.get("arrears_count"),
        )
        return {
            "business_facts": facts,
            "trace": [*state.get("trace", []), "arrears_check_flow"],
        }

    async def _node_fee_verify_flow(state: HybridGraphState) -> HybridGraphState:
        logger.info("hybrid[{}] node=fee_verify_flow start", request_id)
        facts = await fee_facts_fn(state["payload"])
        logger.info(
            "hybrid[{}] node=fee_verify_flow amount_check_result={} error={}",
            request_id,
            facts.get("amount_check_result"),
            facts.get("error"),
        )
        return {
            "business_facts": facts,
            "trace": [*state.get("trace", []), "fee_verify_flow"],
        }

    async def _node_rag_retrieve(state: HybridGraphState) -> HybridGraphState:
        logger.info("hybrid[{}] node=rag_retrieve start", request_id)
        items = await retrieve_fn(state["payload"])
        logger.info("hybrid[{}] node=rag_retrieve retrieved_count={}", request_id, len(items))
        return {
            "retrieved_items": items,
            "trace": [*state.get("trace", []), f"rag_retrieve:{len(items)}"],
        }

    async def _node_answer_synthesizer(state: HybridGraphState) -> HybridGraphState:
        items = state.get("retrieved_items", [])
        facts = state.get("business_facts", {})
        intent = state.get("intent", "rule_explain")
        logger.info(
            "hybrid[{}] node=answer_synthesizer start intent={} retrieved_count={} facts_keys={}",
            request_id,
            intent,
            len(items),
            sorted(facts.keys()),
        )

        if not items and not facts:
            logger.info("hybrid[{}] node=answer_synthesizer no_data", request_id)
            return {
                "conclusion": "未检索到可用证据，暂时无法回答该问题。",
                "key_points": [],
                "model": "",
                "trace": [*state.get("trace", []), "answer_synthesizer:no_data"],
            }

        conclusion, key_points, model_used = await synthesize_fn(
            state["payload"].query,
            items,
            facts,
            intent,
        )
        logger.info(
            "hybrid[{}] node=answer_synthesizer done model={} key_points={}",
            request_id,
            model_used,
            len(key_points),
        )
        return {
            "conclusion": conclusion,
            "key_points": key_points,
            "model": model_used,
            "trace": [*state.get("trace", []), "answer_synthesizer"],
        }

    graph = StateGraph(HybridGraphState)
    graph.add_node("intent_classifier", _node_intent_classifier)
    graph.add_node("rule_explain_flow", _node_rule_explain_flow)
    graph.add_node("arrears_check_flow", _node_arrears_check_flow)
    graph.add_node("fee_verify_flow", _node_fee_verify_flow)
    graph.add_node("rag_retrieve", _node_rag_retrieve)
    graph.add_node("answer_synthesizer", _node_answer_synthesizer)
    graph.set_entry_point("intent_classifier")
    graph.add_conditional_edges(
        "intent_classifier",
        _route_from_intent,
        {
            "rule_explain_flow": "rule_explain_flow",
            "arrears_check_flow": "arrears_check_flow",
            "fee_verify_flow": "fee_verify_flow",
        },
    )
    graph.add_edge("rule_explain_flow", "rag_retrieve")
    graph.add_edge("fee_verify_flow", "rag_retrieve")
    graph.add_edge("arrears_check_flow", "answer_synthesizer")
    graph.add_edge("rag_retrieve", "answer_synthesizer")
    graph.add_edge("answer_synthesizer", END)
    app = graph.compile()
    return await app.ainvoke({"payload": payload, "trace": []})
