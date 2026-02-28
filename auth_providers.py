"""
auth_providers.py — OAuth2 authentication for Gemini (Google) and OpenAI setup.

Google OAuth2 flow:
  1. GET  /auth/google/start          → redirect URL to Google consent screen
  2. GET  /auth/google/callback?code= → exchange code for tokens, store in .env
  3. GET  /auth/google/status         → check if connected
  4. POST /auth/google/revoke         → disconnect

OpenAI key setup (no OAuth — Plus ≠ API):
  1. GET  /auth/openai/status         → check current key status
  2. POST /auth/openai/setup          → validate + save key
  3. GET  /auth/openai/guide          → step-by-step instructions
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["auth"])

HERE = Path(__file__).parent
ENV_FILE = HERE / ".env"

# ─── helpers ──────────────────────────────────────────────────────────────────

def _read_env() -> Dict[str, str]:
    from dotenv import dotenv_values
    return dict(dotenv_values(ENV_FILE)) if ENV_FILE.exists() else {}

def _write_env(updates: Dict[str, str]):
    from dotenv import load_dotenv
    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
    for k, v in updates.items():
        found = False
        for i, line in enumerate(lines):
            if line.startswith(f"{k}=") or line.startswith(f"{k} ="):
                lines[i] = f"{k}={v}"
                found = True
                break
        if not found:
            lines.append(f"{k}={v}")
    ENV_FILE.write_text("\n".join(lines) + "\n")
    load_dotenv(ENV_FILE, override=True)


# ════════════════════════════════════════════════════════════════════════════
# GOOGLE OAUTH2 — Gemini
# ════════════════════════════════════════════════════════════════════════════

GOOGLE_AUTH_URL   = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL  = "https://oauth2.googleapis.com/token"
GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"
GOOGLE_USERINFO   = "https://www.googleapis.com/oauth2/v3/userinfo"

# Scopes: Gemini API + user profile (so we can show who's logged in)
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/generative-language",
    "openid",
    "email",
    "profile",
]


def _google_creds() -> Dict[str, str]:
    env = _read_env()
    return {
        "client_id":     env.get("GOOGLE_CLIENT_ID", ""),
        "client_secret": env.get("GOOGLE_CLIENT_SECRET", ""),
        "access_token":  env.get("GOOGLE_ACCESS_TOKEN", ""),
        "refresh_token": env.get("GOOGLE_REFRESH_TOKEN", ""),
        "token_expiry":  env.get("GOOGLE_TOKEN_EXPIRY", "0"),
        "user_email":    env.get("GOOGLE_USER_EMAIL", ""),
        "user_name":     env.get("GOOGLE_USER_NAME", ""),
    }


async def _refresh_google_token(client_id: str, client_secret: str, refresh_token: str) -> Optional[Dict]:
    """Use refresh_token to get a new access_token."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(GOOGLE_TOKEN_URL, data={
            "grant_type":    "refresh_token",
            "refresh_token": refresh_token,
            "client_id":     client_id,
            "client_secret": client_secret,
        })
        if r.status_code == 200:
            return r.json()
    return None


@router.get("/google/status")
async def google_status():
    """Return current Google auth state."""
    creds = _google_creds()
    configured = bool(creds["client_id"] and creds["client_secret"])
    connected  = bool(creds["access_token"] or creds["refresh_token"])

    # Check if token needs refresh
    token_valid = False
    if connected and creds["refresh_token"]:
        expiry = float(creds["token_expiry"] or 0)
        token_valid = expiry > time.time() + 60  # 1-min buffer
        if not token_valid and creds["client_id"]:
            # Try refresh in background
            refreshed = await _refresh_google_token(
                creds["client_id"], creds["client_secret"], creds["refresh_token"]
            )
            if refreshed:
                expiry = time.time() + refreshed.get("expires_in", 3600)
                _write_env({
                    "GOOGLE_ACCESS_TOKEN": refreshed.get("access_token", ""),
                    "GOOGLE_TOKEN_EXPIRY": str(int(expiry)),
                })
                token_valid = True

    return {
        "provider":      "google",
        "configured":    configured,  # client_id + secret set
        "connected":     connected and token_valid,
        "user_email":    creds["user_email"],
        "user_name":     creds["user_name"],
        "has_client_id": bool(creds["client_id"]),
        "setup_needed":  not configured,
    }


class GoogleClientReq(BaseModel):
    client_id: str
    client_secret: str


@router.post("/google/credentials")
async def save_google_credentials(req: GoogleClientReq):
    """Save Google OAuth2 client_id and client_secret."""
    _write_env({
        "GOOGLE_CLIENT_ID":     req.client_id.strip(),
        "GOOGLE_CLIENT_SECRET": req.client_secret.strip(),
    })
    return {"saved": True, "message": "Google OAuth credentials saved. Now click 'Sign in with Google'."}


@router.get("/google/start")
async def google_oauth_start(request: Request):
    """Generate the Google OAuth2 authorization URL."""
    creds = _google_creds()
    if not creds["client_id"] or not creds["client_secret"]:
        raise HTTPException(400, "Google OAuth not configured. Add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET first.")

    # Callback URL — use request host so it works on any server
    callback = str(request.base_url).rstrip("/") + "/auth/google/callback"

    from urllib.parse import urlencode
    params = {
        "client_id":     creds["client_id"],
        "redirect_uri":  callback,
        "response_type": "code",
        "scope":         " ".join(GOOGLE_SCOPES),
        "access_type":   "offline",   # gets refresh_token
        "prompt":        "consent",   # force consent to always get refresh_token
    }
    url = f"{GOOGLE_AUTH_URL}?{urlencode(params)}"
    return {"url": url, "redirect_uri": callback}


@router.get("/google/callback")
async def google_oauth_callback(request: Request, code: str = None, error: str = None):
    """Handle Google OAuth2 callback, exchange code for tokens."""
    if error:
        return HTMLResponse(f"""
        <html><body style="font-family:sans-serif;background:#0a0a0a;color:#ef4444;padding:40px">
        <h2>Auth failed: {error}</h2>
        <script>setTimeout(()=>window.close(),3000)</script>
        </body></html>""")

    if not code:
        raise HTTPException(400, "No code received from Google.")

    creds = _google_creds()
    callback = str(request.base_url).rstrip("/") + "/auth/google/callback"

    # Exchange code for tokens
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(GOOGLE_TOKEN_URL, data={
            "code":          code,
            "client_id":     creds["client_id"],
            "client_secret": creds["client_secret"],
            "redirect_uri":  callback,
            "grant_type":    "authorization_code",
        })
        if r.status_code != 200:
            raise HTTPException(400, f"Token exchange failed: {r.text[:200]}")
        tokens = r.json()

    # Fetch user info
    user_info = {}
    access_token = tokens.get("access_token", "")
    if access_token:
        async with httpx.AsyncClient(timeout=10) as c:
            ui = await c.get(GOOGLE_USERINFO, headers={"Authorization": f"Bearer {access_token}"})
            if ui.status_code == 200:
                user_info = ui.json()

    expiry = int(time.time() + tokens.get("expires_in", 3600))
    _write_env({
        "GOOGLE_ACCESS_TOKEN":  access_token,
        "GOOGLE_REFRESH_TOKEN": tokens.get("refresh_token", creds.get("refresh_token", "")),
        "GOOGLE_TOKEN_EXPIRY":  str(expiry),
        "GOOGLE_USER_EMAIL":    user_info.get("email", ""),
        "GOOGLE_USER_NAME":     user_info.get("name", ""),
        # Also set GEMINI_API_KEY sentinel so the provider system knows Gemini is active
        "GEMINI_AUTH_METHOD":   "oauth",
    })

    name = user_info.get("name", "you")
    email = user_info.get("email", "")

    return HTMLResponse(f"""
    <html><head><title>Connected!</title></head>
    <body style="font-family:sans-serif;background:#0a0a0a;color:#22c55e;padding:60px;text-align:center">
      <div style="font-size:48px;margin-bottom:16px">✓</div>
      <h2 style="color:#fff;margin:0">Connected as {name}</h2>
      <p style="color:#6b7280">{email}</p>
      <p style="color:#6b7280;margin-top:24px">Gemini is ready. You can close this window.</p>
      <script>
        // Notify the opener that auth completed
        if (window.opener) {{
          window.opener.postMessage({{type:'google_auth_complete',email:'{email}',name:'{name}'}}, '*');
          setTimeout(()=>window.close(), 2000);
        }}
      </script>
    </body></html>""")


@router.post("/google/revoke")
async def google_revoke():
    """Disconnect Google account."""
    creds = _google_creds()
    if creds["access_token"]:
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                await c.post(GOOGLE_REVOKE_URL, params={"token": creds["access_token"]})
        except Exception:
            pass
    _write_env({
        "GOOGLE_ACCESS_TOKEN":  "",
        "GOOGLE_REFRESH_TOKEN": "",
        "GOOGLE_TOKEN_EXPIRY":  "0",
        "GOOGLE_USER_EMAIL":    "",
        "GOOGLE_USER_NAME":     "",
        "GEMINI_AUTH_METHOD":   "",
    })
    return {"disconnected": True}


@router.get("/google/test")
async def google_test():
    """Make a real Gemini API call using OAuth credentials to verify everything works."""
    creds = _google_creds()
    if not creds["access_token"] and not creds["refresh_token"]:
        return {"ok": False, "error": "Not connected. Sign in with Google first."}

    # Refresh if needed
    expiry = float(creds["token_expiry"] or 0)
    if expiry < time.time() + 60 and creds["refresh_token"]:
        refreshed = await _refresh_google_token(
            creds["client_id"], creds["client_secret"], creds["refresh_token"]
        )
        if refreshed:
            creds["access_token"] = refreshed["access_token"]
            _write_env({"GOOGLE_ACCESS_TOKEN": creds["access_token"],
                        "GOOGLE_TOKEN_EXPIRY": str(int(time.time() + refreshed.get("expires_in", 3600)))})

    try:
        start = time.time()
        from google.oauth2.credentials import Credentials
        from google import genai as google_genai

        credentials = Credentials(
            token=creds["access_token"],
            refresh_token=creds["refresh_token"],
            token_uri=GOOGLE_TOKEN_URL,
            client_id=creds["client_id"],
            client_secret=creds["client_secret"],
        )
        client = google_genai.Client(credentials=credentials)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Say 'Gemini connected via OAuth' in exactly those words."
        )
        latency = round((time.time() - start) * 1000)
        return {
            "ok": True,
            "latency_ms": latency,
            "response": response.text,
            "model": "gemini-2.0-flash",
            "auth": "oauth2",
            "user": creds["user_email"],
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


# ════════════════════════════════════════════════════════════════════════════
# OPENAI — API key setup (Plus ≠ API, guided flow)
# ════════════════════════════════════════════════════════════════════════════

@router.get("/openai/status")
async def openai_status():
    """Return current OpenAI key status with account type detection."""
    env = _read_env()
    key = env.get("OPENAI_API_KEY", "")

    if not key:
        return {
            "configured": False,
            "has_key": False,
            "plus_note": True,
            "message": "No API key set. ChatGPT Plus does not include API access — see setup guide.",
        }

    # Validate the key with a real request
    try:
        start = time.time()
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get("https://api.openai.com/v1/models",
                           headers={"Authorization": f"Bearer {key}"})
        latency = round((time.time() - start) * 1000)

        if r.status_code == 200:
            data = r.json()
            models = [m["id"] for m in data.get("data", []) if "gpt" in m["id"]]
            # Detect tier by available models
            has_o_series = any("o4" in m or "o3" in m for m in models)
            has_gpt4 = any("gpt-4" in m for m in models)

            return {
                "configured": True,
                "has_key": True,
                "valid": True,
                "latency_ms": latency,
                "model_count": len(models),
                "has_gpt4": has_gpt4,
                "has_o_series": has_o_series,
                "masked_key": key[:8] + "..." + key[-4:],
                "tier": "tier-1+" if has_o_series else ("pay-as-you-go" if has_gpt4 else "free-trial"),
            }
        elif r.status_code == 401:
            return {"configured": True, "has_key": True, "valid": False,
                    "error": "Invalid API key — check it at platform.openai.com"}
        elif r.status_code == 429:
            return {"configured": True, "has_key": True, "valid": True,
                    "latency_ms": latency, "note": "Rate limited — key is valid but quota exceeded"}
        else:
            return {"configured": True, "has_key": True, "valid": False,
                    "error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"configured": True, "has_key": True, "valid": False, "error": str(e)[:100]}


class OpenAIKeyReq(BaseModel):
    api_key: str


@router.post("/openai/setup")
async def openai_setup(req: OpenAIKeyReq):
    """Validate and save an OpenAI API key."""
    key = req.api_key.strip()
    if not key.startswith("sk-"):
        raise HTTPException(400, "Invalid key format — OpenAI API keys start with 'sk-'")

    # Validate before saving
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get("https://api.openai.com/v1/models",
                       headers={"Authorization": f"Bearer {key}"})

    if r.status_code == 401:
        raise HTTPException(401, "Invalid API key. Make sure you copied it from platform.openai.com/api-keys")
    if r.status_code not in (200, 429):
        raise HTTPException(400, f"Unexpected response: HTTP {r.status_code}")

    _write_env({"OPENAI_API_KEY": key})
    models = [m["id"] for m in r.json().get("data", [])] if r.status_code == 200 else []
    gpt4_count = sum(1 for m in models if "gpt-4" in m)

    return {
        "saved": True,
        "valid": True,
        "model_count": len(models),
        "gpt4_models": gpt4_count,
        "message": f"OpenAI connected — {len(models)} models available including {gpt4_count} GPT-4 variants.",
    }


@router.get("/openai/guide")
async def openai_guide():
    """Step-by-step guide for getting an OpenAI API key from a Plus account."""
    return {
        "title": "Connect OpenAI API (ChatGPT Plus account)",
        "important_note": {
            "title": "ChatGPT Plus ≠ API Access",
            "body": "Your $20/mo ChatGPT Plus subscription is for chat.openai.com only. "
                    "The API has separate billing — but you use the SAME OpenAI account. "
                    "You only pay for what you use (GPT-4o Mini costs $0.15 per million tokens).",
        },
        "steps": [
            {
                "step": 1,
                "title": "Go to API Keys",
                "action": "open_url",
                "url": "https://platform.openai.com/api-keys",
                "description": "Log in with your existing OpenAI account (same one as ChatGPT Plus).",
            },
            {
                "step": 2,
                "title": "Create a new key",
                "description": "Click '+ Create new secret key'. Give it a name like 'AgentFlow'. Copy it immediately — it's only shown once.",
            },
            {
                "step": 3,
                "title": "Add billing (if first time)",
                "action": "open_url",
                "url": "https://platform.openai.com/settings/organization/billing",
                "description": "First-time API users need to add a payment method. Minimum $5 top-up. "
                               "GPT-4o Mini costs ~$0.15/1M tokens — $5 lasts a very long time.",
            },
            {
                "step": 4,
                "title": "Paste your key below",
                "description": "Paste the sk-... key into the field and click Connect.",
            },
        ],
        "cost_estimate": {
            "gpt_4o_mini": "$0.15 / 1M input tokens — cheapest, fast",
            "gpt_4o":      "$5.00 / 1M input tokens — best quality",
            "o4_mini":     "$1.10 / 1M input tokens — best reasoning",
        },
    }
