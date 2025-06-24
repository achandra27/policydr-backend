from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Allow all origins (for now, can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    query: str
    session_id: str

@app.get("/")
def read_root():
    return {"message": "PolicyDr Agentic AI backend is running."}

@app.post("/upload")
async def upload_policy(file: UploadFile = File(...)):
    # TEMP: Just return filename for now
    contents = await file.read()
    return {"filename": file.filename, "size": len(contents)}

@app.post("/ask")
async def ask_question(request: AskRequest):
    # TEMP: Dummy logic for now
    return {
        "answer": f"You asked: '{request.query}' for session '{request.session_id}'. AI logic goes here."
    }

# Local test
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
