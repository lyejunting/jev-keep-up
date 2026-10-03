from contextvars import ContextVar
from time import perf_counter

from fastapi.routing import APIRoute


# AnyIO propagates this request-local dictionary into FastAPI's worker thread.
prediction_timings: ContextVar[dict[str, float] | None] = ContextVar("prediction_timings", default=None)
STAGES = ("request_parsing", "premise", "tokenization", "model_forward", "post_processing", "total")


class PredictTimingRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def timed_handler(request):
            timings = dict.fromkeys(STAGES, 0.0)
            token = prediction_timings.set(timings)
            started = perf_counter()
            body = request.body
            json = request.json

            async def timed_body():
                before = perf_counter()
                try:
                    return await body()
                finally:
                    timings["request_parsing"] += (perf_counter() - before) * 1000

            async def timed_json():
                before = perf_counter()
                prior = timings["request_parsing"]
                try:
                    return await json()
                finally:
                    # Request.json() may read the body; count that time once.
                    timings["request_parsing"] += (perf_counter() - before) * 1000 - (timings["request_parsing"] - prior)

            request.body = timed_body
            request.json = timed_json
            try:
                response = await handler(request)
                timings["total"] = (perf_counter() - started) * 1000
                response.headers["Server-Timing"] = ", ".join(f"{stage};dur={timings[stage]:.3f}" for stage in STAGES)
                return response
            finally:
                request.body = body
                request.json = json
                prediction_timings.reset(token)

        return timed_handler
