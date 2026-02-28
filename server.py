from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import base64
from google import genai
from google.genai import types

app = FastAPI()

API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None

TEXT_MODEL = "gemini-2.5-flash"
IMAGE_MODEL = "gemini-2.5-flash" # Используем основную модель, она умеет в картинки
KEY_COLOR = "#00FF00"

# --- Схемы ---
class TextRequest(BaseModel):
    prompt: str

class ImageRequest(BaseModel):
    prompt: str

class LayerRequest(BaseModel):
    prompt: str
    layer_name: str
    layer_kind: str
    key_color: str | None = None

# Схема для структурированного JSON ответа
class LayerItem(BaseModel):
    name: str
    role: str
    kind: str
    prompt: str

class DecomposeResponse(BaseModel):
    layers: list[LayerItem]

# --- Роуты ---
@app.get("/")
def root():
    # Читаем HTML прямо из корня (удобно для деплоя с телефона)
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/health")
def health():
    return {
        "ok": True,
        "has_key": bool(API_KEY),
        "key_color": KEY_COLOR,
    }

def _extract_image_b64(response):
    try:
        # Пытаемся вытащить base64 из ответа
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.inline_data and part.inline_data.data:
                    data = part.inline_data.data
                    if isinstance(data, (bytes, bytearray)):
                        return base64.b64encode(data).decode("utf-8")
                    return data
    except Exception:
        pass
    return None

@app.post("/decompose_scene")
def decompose_scene(req: TextRequest):
    if not client:
        return {"error": "Нет API ключа"}
    try:
        prompt = f"Разбей сцену на слои: один 'background', остальные 'object'. Сцена: {req.prompt}"
        resp = client.models.generate_content(
            model=TEXT_MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DecomposeResponse,
            ),
        )
        return resp.text
    except Exception as e:
        return {"error": str(e)}

@app.post("/generate_layer")
def generate_layer(req: LayerRequest):
    if not client:
        return {"error": "Нет API ключа"}
    
    key_color = req.key_color or KEY_COLOR
    
    if req.layer_kind == "background":
        final_prompt = f"{req.prompt}\nRules: Background only, no main subjects, highly detailed."
    else:
        final_prompt = f"{req.prompt}\nSTRICT RULES: One single object. Background MUST be perfectly solid {key_color} with NO shadows and NO gradients."

    try:
        resp = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[final_prompt]
        )
        img_b64 = _extract_image_b64(resp)
        if img_b64:
            return {"image_base64": img_b64, "mime_type": "image/jpeg", "key_color": key_color}
        return {"error": "Модель не вернула картинку"}
    except Exception as e:
        return {"error": str(e)}
