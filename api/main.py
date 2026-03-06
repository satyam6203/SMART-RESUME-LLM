import os
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "resume_llm")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)


@app.get("/generate")
def generate_resume(description: str):

    inputs = tokenizer(description, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=200
    )

    result = tokenizer.decode(outputs[0])

    return {"resume": result}