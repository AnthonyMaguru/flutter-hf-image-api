import os
from pathlib import Path
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY")

app = FastAPI(title="Flutter Pollinations Image Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ImageRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=1000)


@app.get("/")
def home():
    return {
        "message": "Pollinations image API is running",
        "api_key_configured": bool(POLLINATIONS_API_KEY),
    }


@app.post("/generate-image")
def generate_image(request: ImageRequest):
    if not POLLINATIONS_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="POLLINATIONS_API_KEY is missing from the backend .env file.",
        )

    prompt = quote(request.prompt, safe="")
    url = f"https://gen.pollinations.ai/image/{prompt}"

    try:
        with httpx.Client(timeout=180.0) as client:
            upstream = client.get(
                url,
                params={"model": "black-forest-labs/flux.1-schnell"},
                headers={
                    "Authorization": f"Bearer {POLLINATIONS_API_KEY}",
                    "Accept": "image/*",
                },
            )

        if upstream.status_code != 200:
            # Log upstream details locally; never log your API key.
            print(
                f"Pollinations error: HTTP {upstream.status_code}: "
                f"{upstream.text[:1000]}",
                flush=True,
            )

            if upstream.status_code in (401, 402, 403, 429):
                raise HTTPException(
                    status_code=upstream.status_code,
                    detail=(
                        "Pollinations rejected the request. Check your API key, "
                        "permissions, Pollen balance, and rate limits."
                    ),
                )

            raise HTTPException(
                status_code=502,
                detail="Pollinations could not generate the image. Try again later.",
            )

        content_type = upstream.headers.get("content-type", "").split(";")[0]

        if not content_type.startswith("image/"):
            print(
                f"Unexpected Pollinations response type: {content_type}",
                flush=True,
            )
            raise HTTPException(
                status_code=502,
                detail="Pollinations returned a response that was not an image.",
            )

        return Response(
            content=upstream.content,
            media_type=content_type,
            headers={"Cache-Control": "no-store"},
        )

    except HTTPException:
        raise
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Image generation timed out. Please try again.",
        )
    except httpx.HTTPError as error:
        print(f"Pollinations connection error: {error}", flush=True)
        raise HTTPException(
            status_code=502,
            detail="Could not connect to Pollinations. Check your internet connection.",
        )


@app.get("/balance")
def check_balance():
    """Optional balance check. The key must have account:usage permission."""
    if not POLLINATIONS_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="POLLINATIONS_API_KEY is missing.",
        )

    try:
        response = httpx.get(
            "https://gen.pollinations.ai/account/balance",
            headers={"Authorization": f"Bearer {POLLINATIONS_API_KEY}"},
            timeout=20.0,
        )

        if response.status_code != 200:
            print(
                f"Balance check failed: HTTP {response.status_code}: "
                f"{response.text[:500]}",
                flush=True,
            )
            raise HTTPException(
                status_code=response.status_code,
                detail=(
                    "Balance lookup failed. Ensure your key has the "
                    "account:usage permission."
                ),
            )

        return response.json()

    except httpx.HTTPError as error:
        print(f"Balance connection error: {error}", flush=True)
        raise HTTPException(
            status_code=502,
            detail="Could not retrieve your Pollinations balance.",
        )
