from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy


app = FastAPI()

# Download the model at runtime
spacy.cli.download("en_core_web_sm")

# Load the spacy model
nlp = spacy.load("en_core_web_sm")

class TextRequest(BaseModel):
    text: str





@app.get("/")
async def welcome():
    return "Welcome to Spacy Backend API"

@app.post("/analyze")
async def analyze_text(request: TextRequest):
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    doc = nlp(text)
    result = [
        {
            "text": token.text,
            "lemma": token.lemma_,
            "pos": spacy.explain(token.tag_),
            "dependency": spacy.explain(token.dep_),
            "head": token.head.text
        }
        for token in doc
    ]

    return {"result": result}
