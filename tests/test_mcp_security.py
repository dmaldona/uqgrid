import asyncio

import pytest

mcp_package = pytest.importorskip("mcp")

from uqgrid.mcp.security import StaticTokenVerifier


TOKEN = "a-static-test-token-that-is-longer-than-32-characters"


def test_static_token_verifier_accepts_exact_token():
    verifier = StaticTokenVerifier(TOKEN, "https://uqgrid.example/mcp", "alice")
    accepted = asyncio.run(verifier.verify_token(TOKEN))

    assert accepted.subject == "alice"
    assert accepted.resource == "https://uqgrid.example/mcp"
    assert accepted.scopes == ["uqgrid"]


def test_static_token_verifier_rejects_other_token():
    verifier = StaticTokenVerifier(TOKEN, "https://uqgrid.example/mcp", "alice")
    assert asyncio.run(verifier.verify_token("wrong")) is None


def test_static_token_requires_sufficient_entropy():
    with pytest.raises(ValueError, match="at least 32"):
        StaticTokenVerifier("short", "https://uqgrid.example/mcp", "alice")
