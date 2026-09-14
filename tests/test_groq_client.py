import importlib


def test_call_llm_retries_without_response_format_on_json_validation_error(monkeypatch):
    module = importlib.import_module("core.llm_client")

    class FakeMessage:
        content = '{"recommendation": "Keep saving", "reason": "Test", "confidence": 0.9}'

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise Exception("json_validate_failed")
            return FakeResponse()

    class FakeClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    fake_client = FakeClient()
    monkeypatch.setattr(module, "get_client", lambda: fake_client)

    result = module.call_llm("system", "user", temperature=0.4)

    assert result == {
        "recommendation": "Keep saving",
        "reason": "Test",
        "confidence": 0.9,
    }
    assert len(fake_client.chat.completions.calls) == 2
    assert "response_format" in fake_client.chat.completions.calls[0]
    assert "response_format" not in fake_client.chat.completions.calls[1]
