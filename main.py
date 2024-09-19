from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Download the model at runtime
spacy.cli.download("en_core_web_sm")

# Load the spacy model
nlp = spacy.load("en_core_web_sm")

class TextRequest(BaseModel):
    text: str

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
