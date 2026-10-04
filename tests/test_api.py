from fastapi.testclient import TestClient

from kairos.api import app


def test_health_and_replay_and_websocket():
    with TestClient(app) as client:
        h = client.get("/health").json()
        assert h["status"] == "ok" and h["index"]["chunks"] > 100
        r = client.post("/v1/replay", json={"turns": [{"utterance": "How many days of sick leave do we get?"}]}).json()
        assert r["turns"][0]["citations"] == ["Doc_19 §2"]
        with client.websocket_connect("/v1/stream") as ws:
            assert ws.receive_json()["type"] == "session"
            ws.send_json({"type": "chunk", "t": 0.0, "text": "What is the kitchen access fee"})
            assert ws.receive_json()["type"] == "events"
            ws.send_json({"type": "chunk", "t": 0.9, "text": "at The Deccan Courtyard?"})
            assert ws.receive_json()["type"] == "events"
            ws.send_json({"type": "end", "t": 1.5})
            msgs = []
            while True:
                m = ws.receive_json()
                msgs.append(m)
                if m["type"] == "record":
                    break
            rec = msgs[-1]["record"]
            assert "₹5,000" in rec["answer"] and rec["telemetry"]["early_retrieval"]
