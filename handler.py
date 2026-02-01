import runpod
import torch
from diffusers import StableDiffusionXLPipeline
import base64
from io import BytesIO

# 1. Cargar el modelo (SDXL Turbo es ultra rápido)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/sdxl-turbo", 
    torch_dtype=torch.float16, 
    variant="fp16"
)
pipe.to("cuda")

def handler(event):
    # Leer el prompt que enviamos desde la App
    input_data = event.get("input", {})
    prompt = input_data.get("prompt", "A high quality game mission thumbnail")

    # 2. Generar la imagen
    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

    # 3. Convertir imagen a Base64 para enviarla de vuelta
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"image_base64": img_str}

runpod.serverless.start({"handler": handler})
