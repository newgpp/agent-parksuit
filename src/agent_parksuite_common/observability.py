from __future__ import annotations

import sys
from contextvars import ContextVar, Token
from uuid import uuid4

from fastapi import Request
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

TRACE_ID_HEADER = "X-Trace-Id"

trace_id_ctx: ContextVar[str] = ContextVar("trace_id", default="-")


def setup_loguru(service_name: str) -> None:
    logger.remove()
    logger.configure(
        extra={"service": service_name},
        patcher=lambda record: record["extra"].update(
            {
                "trace_id": trace_id_ctx.get("-"),
            }
        ),
    )
    logger.add(
        sys.stdout,
        level="INFO",
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {extra[service]} "
            "| trace_id={extra[trace_id]} | {message}"
        ),
    )


def current_trace_headers() -> dict[str, str]:
    return {
        TRACE_ID_HEADER: trace_id_ctx.get("-"),
    }


class TraceContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id = request.headers.get(TRACE_ID_HEADER) or uuid4().hex
        trace_token: Token[str] = trace_id_ctx.set(trace_id)

        logger.info("request.start method={} path={}", request.method, request.url.path)
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request.error method={} path={}", request.method, request.url.path)
            raise
        else:
            response.headers[TRACE_ID_HEADER] = trace_id
            logger.info(
                "request.end method={} path={} status_code={}",
                request.method,
                request.url.path,
                response.status_code,
            )
            return response
        finally:
            trace_id_ctx.reset(trace_token)
