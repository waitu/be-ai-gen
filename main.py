from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google_auth_oauthlib.flow import Flow
import os
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


IMGBB_API_KEY = "94fa8f8b2953e385921d3f3d84370e23"  # Thay bằng API key imgbb của bạn

REPLICATE_MODEL_VERSION = "lucataco/sdxl-controlnet:06d6fae3b75ab68a28cd2900afa6033166910dd09fd9751047043a5bbb4c184b"


def upload_to_imgbb(image_path):
    with open(image_path, "rb") as file:
        res = requests.post(
            "https://api.imgbb.com/1/upload",
            data={"key": IMGBB_API_KEY},
            files={"image": file},
        )
    if res.status_code == 200:
        return res.json()["data"]["url"]
    else:
        return None


@app.post("/generate")
async def generate_image(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    condition_scale: float = Form(0.5),
    negative_prompt: str = Form("low quality, bad quality, sketches"),
    num_inference_steps: int = Form(50),
    replicate_model_version: str = Form(...),
):
    # Lưu file tạm
    temp_path = f"temp_{image.filename}"
    with open(temp_path, "wb") as f:
        f.write(await image.read())

    # Upload ảnh lên imgbb để lấy URL
    image_url = upload_to_imgbb(temp_path)
    os.remove(temp_path)
    if not image_url:
        return JSONResponse(
            {"error": "Không upload được ảnh lên imgbb"}, status_code=400
        )

    # Gọi API Replicate
    response = requests.post(
        "https://api.replicate.com/v1/predictions",
        headers={
            "Authorization": f"Bearer r8_bGaSHGCkmoIWW5eWcrjsxWVQLretU5G3XicUH",
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

    # Tải ảnh từ image_result_url về tạm
    import tempfile

    img_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    img_data = requests.get(image_result_url).content
    img_temp.write(img_data)
    img_temp.close()

    # Upload lại lên imgbb
    imgbb_url = upload_to_imgbb(img_temp.name)
    os.remove(img_temp.name)

    if not imgbb_url:
        return JSONResponse(
            {"error": "Không upload được ảnh lên imgbb"}, status_code=400
        )

    return {"imgbb_url": imgbb_url}
