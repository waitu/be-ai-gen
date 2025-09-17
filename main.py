from fastapi import FastAPI, Request, File, UploadFile, Form, Header
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google_auth_oauthlib.flow import Flow
import os
import requests
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


REPLICATE_MODEL_VERSION = "lucataco/sdxl-controlnet:06d6fae3b75ab68a28cd2900afa6033166910dd09fd9751047043a5bbb4c184b"


def poll_replicate_prediction(get_url, authorization, max_wait=60):
    waited = 0
    while waited < max_wait:
        resp = requests.get(get_url, headers={"Authorization": authorization})
        data = resp.json()
        if data.get("status") == "succeeded" and data.get("output"):
            return data
        if data.get("status") == "failed":
            return data
        time.sleep(2)
        waited += 2
    return data  # timeout


@app.post("/api/generate")
async def generate_image(
    image: UploadFile = File(None),
    image_url: str = Form(None),
    prompt: str = Form(...),
    condition_scale: float = Form(0.5),
    negative_prompt: str = Form("low quality, bad quality, sketches"),
    num_inference_steps: int = Form(50),
    replicate_model_version: str = Form(...),
    authorization: str = Header(None),
):
    # This endpoint now expects a public `image_url` (preferred). If a file is uploaded instead,
    # the server will ask the client to provide a public URL because hosting files is out of scope.
    if not authorization:
        return JSONResponse({"error": "Thiếu Authorization header"}, status_code=400)

    if not image_url:
        return JSONResponse({"error": "Vui lòng cung cấp một image_url công khai trong trường 'image_url' hoặc gửi một URL thay vì file"}, status_code=400)

    # Call Replicate
    response = requests.post(
        "https://api.replicate.com/v1/predictions",
        headers={
            "Authorization": authorization,
            "Content-Type": "application/json",
            "Prefer": "wait",
        },
        json={
            "version": replicate_model_version,
            "input": {
                "image": image_url,
                "prompt": prompt,
                "condition_scale": condition_scale,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
            },
        },
    )
    output = response.json()
    if output.get("status") != "succeeded":
        get_url = output.get("urls", {}).get("get")
        if get_url:
            output = poll_replicate_prediction(get_url, authorization)

    image_output = output.get("output")
    if isinstance(image_output, list):
        image_result_url = image_output[0] if image_output else None
    elif isinstance(image_output, str):
        image_result_url = image_output
    else:
        image_result_url = None

    if not image_result_url:
        return JSONResponse(
            {
                "error": "Không nhận được link ảnh từ Replicate",
                "replicate_output": output,
            },
            status_code=400,
        )

    return {"prompt": prompt, "replicate_image_url": image_result_url}


@app.get("/")
async def home():
    return {"ok"}


@app.post("/api/generate-text2img")
async def generate_text2img(
    prompt: str = Form(...),
    replicate_model_version: str = Form(...),
    aspect_ratio: str = Form("1:1"),
    output_format: str = Form("png"),
    output_quality: int = Form(50),
    safety_tolerance: int = Form(2),
    prompt_upsampling: bool = Form(True),
    authorization: str = Header(None),
):
    if not authorization:
        return JSONResponse({"error": "Thiếu Authorization header"}, status_code=400)

    # Gọi API Replicate text-to-image
    response = requests.post(
        "https://api.replicate.com/v1/predictions",
        headers={
            "Authorization": authorization,
            "Content-Type": "application/json",
            "Prefer": "wait",
        },
        json={
            "version": replicate_model_version,
            "input": {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "output_format": output_format,
                "output_quality": output_quality,
                "safety_tolerance": safety_tolerance,
                "prompt_upsampling": prompt_upsampling,
            },
        },
    )
    output = response.json()
    if output.get("status") != "succeeded":
        get_url = output.get("urls", {}).get("get")
        if get_url:
            output = poll_replicate_prediction(get_url, authorization)

    image_output = output.get("output")
    if isinstance(image_output, list):
        image_result_url = image_output[0] if image_output else None
    elif isinstance(image_output, str):
        image_result_url = image_output
    else:
        image_result_url = None

    if not image_result_url:
        return JSONResponse(
            {
                "error": "Không nhận được link ảnh từ Replicate",
                "replicate_output": output,
            },
            status_code=400,
        )

    # Return replicate image URL directly
    return {"prompt": prompt, "imgbb_url": image_result_url, "image_url": image_result_url}
