"""Authentication helpers for controlled UQGrid deployments."""

import hmac

from mcp.server.auth.provider import AccessToken


class StaticTokenVerifier:
    """Verify one pre-shared bearer token for the initial single-user service."""

    def __init__(self, token: str, resource: str, subject: str):
        if len(token) < 32:
            raise ValueError("UQGRID_API_TOKEN must contain at least 32 characters")
        self.token = token
        self.resource = resource
        self.subject = subject

    async def verify_token(self, token: str):
        if not hmac.compare_digest(token, self.token):
            return None
        return AccessToken(
            token=token,
            client_id="uqgrid-static-client",
            scopes=["uqgrid"],
            resource=self.resource,
            subject=self.subject,
        )
