"""One local brain, serialized mutations, no cross-origin browser access."""
from threading import RLock
from typing import Literal
from fastapi import FastAPI, HTTPException, Request, Query
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .brain import FlyBrain


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    ms: float = Field(default=10, ge=0, le=1000)


def create_app(brain=None):
    brain = brain if brain is not None else FlyBrain("malecns")
    app = FastAPI(title="FlyBrain local API", version="0.1.0")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])
    app.state.brain = brain
    lock = RLock()

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        from starlette.responses import JSONResponse
        # Native clients send no Origin; browser integrations use a local proxy.
        if request.headers.get("origin"):
            return JSONResponse({"detail": "Browser cross-origin requests are disabled; use a local application proxy"}, status_code=403)
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            return JSONResponse({"detail": "Invalid content length"}, status_code=400)
        if length > 65536:
            return JSONResponse({"detail": "Request too large"}, status_code=413)
        return await call_next(request)

    def call(function, *args, **kwargs):
        with lock:
            try: return function(*args, **kwargs)
            except (ValueError, TypeError) as error:
                raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/stimulus")
    def stimulus(payload: dict):
        call(brain.stimulus, payload)
        return {"ok": True, "persistence": "until replaced or reset"}

    @app.post("/step")
    def step(payload: StepRequest):
        with lock:
            call(brain.step, payload.ms)
            return {"time_ms": brain.engine.time_ms}

    @app.get("/activity")
    def activity(region: str | None = None, neurons: list[str] | None = Query(default=None),
                 offset: int = Query(default=0, ge=0), limit: int = Query(default=1000, ge=1, le=10000)):
        result = call(brain.activity, region=region, neurons=neurons)
        result["total"] = len(result["neuron_ids"])
        for key in ("neuron_ids", "spikes", "spike_counts", "voltage_mv", "rates_hz"):
            result[key] = result[key][offset:offset+limit]
        result.update(offset=offset, limit=limit)
        return result

    @app.get("/output")
    def output(name: Literal["descending", "motor"] = "descending"):
        return call(brain.output, name)

    @app.post("/reset")
    def reset():
        call(brain.reset)
        return {"ok": True, "time_ms": 0}

    @app.get("/provenance")
    def provenance():
        return brain.provenance

    @app.get("/explain/input/{name}")
    def explain_input(name: str):
        return call(brain.explain_input, name)

    return app
