from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from models import load_models

app = FastAPI()
models = {} 

# static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

class CodeInput(BaseModel):
    code: str
    model: str = "codet5"  # default

@app.post("/summarize")
def summarize(input: CodeInput):
    model_name = input.model.lower()

    if model_name not in models:
        # Load on demand
        if model_name == "codet5":
            tokenizer, model = load_models.load_codet5_ft()
        elif model_name == "codet5_base":
            tokenizer, model = load_models.load_codet5()
        elif model_name == "deepseek":
            tokenizer, model = load_models.load_deepseek_ft()
        elif model_name == "deepseek_base":
            tokenizer, model = load_models.load_deepseek()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model '{model_name}'")

        models[model_name] = {"tokenizer": tokenizer, "model": model}

    tokenizer = models[model_name]["tokenizer"]
    model = models[model_name]["model"]

    inputs = tokenizer(input.code, return_tensors="pt", truncation=True)
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=160,
            num_beams=5,
            early_stopping=True
        )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return {"summary": summary}
