"""A tiny OpenAI-compatible streaming server used to test the LLM path offline. It answers with one
faithful sentence, one sentence with a fabricated citation and one with a wrong number, so the
verifier must keep the first and drop the other two."""
import json
import threading

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    user = body["messages"][-1]["content"]
    first = next(l for l in user.splitlines() if l.startswith("- ["))
    cid = first[3:first.index("]")]
    sentence = first[first.index("]") + 2:]
    reply = (f"{sentence} [{cid}] The company also covers spa treatments every weekend. [Doc_99 §9] "
             f"It costs exactly 123456 rupees. [{cid}]")

    def gen():
        for i in range(0, len(reply), 12):
            yield "data: " + json.dumps({"choices": [{"delta": {"content": reply[i:i + 12]}}]}) + "\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


def serve(port: int) -> threading.Thread:
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(cfg)
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    return th
