import os
import uvicorn
import logging
from fastapi import FastAPI, Request, Response, Depends
from dotenv import load_dotenv
import httpx
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# The URL of llama.cpp server.
LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", "http://llama:8001")

app = FastAPI()


def get_temperature(request: Request):
    """
    Extracts the temperature parameter from the request body.
    If the temperature is not set, defaults to 0.7 and adds 0.3.
    Ensures the minimum temperature is 0.7.
    """
    try:
        body = await request.json()
        temp = max(body.get("temperature", 0.7) + 0.3, 0.7)
        logger.info(f"Extracted temperature: {temp}")
        return temp
    except Exception as e:
        logger.error(f"Error extracting temperature: {e}")
        return 0.7


async def fetch_streamed_responses(request: Request, path: str, temperature: float):
    """
    Sends 4 batched requests to the llama.cpp server in streaming mode.
    Cancels all remaining requests when one completes.
    Returns the first completed response.
    """
    async with httpx.AsyncClient() as client:
        tasks = []  # List to store tasks
        responses = []  # Store successful response
        event = asyncio.Event()  # Event to signal when a response completes
        url = f"{LLAMA_BASE_URL}/{path}"
        body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
        params = request.query_params
        logger.info(f"Sending requests to: {url} with temperature: {temperature}")

        async def make_request():
            """Handles a single request in streaming mode."""
            try:
                async with client.stream("POST", url, headers=headers, params=params, content=body) as response:
                    collected_data = b""
                    async for chunk in response.aiter_bytes():
                        if event.is_set():  # If another response has completed, stop collecting
                            logger.info("Cancelling remaining requests.")
                            return
                        collected_data += chunk
                    responses.append(collected_data)
                    event.set()  # Signal that a response has completed
                    logger.info("Response received, canceling others.")
            except Exception as e:
                logger.error(f"Error during request: {e}")

        # Launch 4 concurrent streaming requests
        for _ in range(4):
            tasks.append(asyncio.create_task(make_request()))

        # Wait for one to complete and cancel others
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Returning first completed response.")
        return Response(content=responses[0], status_code=200) if responses else Response(status_code=500)


@app.api_route("/v1/chat/completions", methods=["POST"])
async def chat_completions(request: Request):
    """
    Handles requests to the /v1/chat/completions endpoint.
    Retrieves temperature, sends 4 batched streaming requests, and returns the first completed response.
    """
    logger.info("Received request for /v1/chat/completions")
    temperature = get_temperature(request)
    return await fetch_streamed_responses(request, "v1/chat/completions", temperature)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy(request: Request, path: str):
    """
    Generic proxy function that forwards any request to the llama.cpp server.
    Handles all HTTP methods dynamically.
    """
    forward_url = f"{LLAMA_BASE_URL}/{path}"
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    logger.info(f"Proxying request to: {forward_url} with method: {request.method}")

    async with httpx.AsyncClient() as client:
        try:
            if request.method == "GET":
                response = await client.get(forward_url, headers=headers, params=request.query_params)
            elif request.method == "POST":
                body = await request.body()
                response = await client.post(forward_url, headers=headers, params=request.query_params, content=body)
            elif request.method == "PUT":
                body = await request.body()
                response = await client.put(forward_url, headers=headers, params=request.query_params, content=body)
            elif request.method == "DELETE":
                response = await client.delete(forward_url, headers=headers, params=request.query_params)
            elif request.method == "PATCH":
                body = await request.body()
                response = await client.patch(forward_url, headers=headers, params=request.query_params, content=body)
            elif request.method == "OPTIONS":
                response = await client.options(forward_url, headers=headers, params=request.query_params)
            elif request.method == "HEAD":
                response = await client.head(forward_url, headers=headers, params=request.query_params)
            else:
                logger.warning("Method not allowed")
                return Response(status_code=405, content="Method Not Allowed")
        except Exception as e:
            logger.error(f"Error during proxy request: {e}")
            return Response(status_code=500, content="Internal Server Error")

    logger.info("Returning proxied response.")
    return Response(content=response.content, status_code=response.status_code, headers=dict(response.headers))


if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

