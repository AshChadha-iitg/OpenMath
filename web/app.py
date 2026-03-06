from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

import inference

app = FastAPI(title="OpenMath API", version="0.1")

# Serve the simple frontend
app.mount("/static", StaticFiles(directory="web/static"), name="static")


@app.get("/ui")
async def ui_index():
    return FileResponse("web/static/index.html")


class SolveRequest(BaseModel):
    problem: str
    cot: Optional[bool] = False
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    max_new_tokens: Optional[int] = 200


class SolveResponse(BaseModel):
    solution: str


@app.get("/", response_model=dict)
async def root():
    return {"message": "OpenMath API. POST /solve with a problem to get a solution."}


@app.post("/solve", response_model=SolveResponse)
async def solve(req: SolveRequest):
    if not req.problem or not req.problem.strip():
        raise HTTPException(status_code=400, detail="`problem` must be a non-empty string")

    try:
        solution = inference.generate_solution(
            problem=req.problem,
            cot=req.cot,
            temperature=req.temperature,
            top_p=req.top_p,
            max_new_tokens=req.max_new_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return SolveResponse(solution=solution)
