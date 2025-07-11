from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict
import os
import uuid

import fitz  # PyMuPDF
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import openai

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Allow CORS (for future frontend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory storage for session-based FAISS indices
session_vectorstores: Dict[str, FAISS] = {}

@app.get("/")
async def root():
    return {"message": "PolicyDr Agentic AI backend is running."}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    contents = await file.read()
    pdf_text = ""
    with fitz.open(stream=contents, filetype="pdf") as doc:
        for page in doc:
            pdf_text += page.get_text()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(pdf_text)

    if not chunks:
        raise HTTPException(status_code=422, detail="No readable text found in the PDF.")

    # Generate embeddings
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    # Save index with a session ID
    session_id = str(uuid.uuid4())
    session_vectorstores[session_id] = vectorstore

    return {
        "message": "File uploaded and indexed successfully.",
        "session_id": session_id,
        "num_chunks": len(chunks)
    }

class AskRequest(BaseModel):
    query: str
    session_id: str

@app.post("/ask")
async def ask_question(req: AskRequest):
    vectorstore = session_vectorstores.get(req.session_id)
    if not vectorstore:
        raise HTTPException(status_code=404, detail="Invalid session_id or session expired.")

    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            retriever=vectorstore.as_retriever()
        )

        answer = qa_chain.run(req.query)
        return {"answer": answer}

    except openai.error.RateLimitError:
        return JSONResponse(
            status_code=200,
            content={
                "answer": "AI temporarily unavailable due to quota limits. Please try again later.",
                "sources": []
            }
        )

    except openai.error.OpenAIError as e:
        return JSONResponse(
            status_code=200,
            content={
                "answer": "Sorry, an error occurred while processing your query.",
                "sources": [],
                "debug": str(e)
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={
                "answer": "An unexpected error occurred.",
                "sources": [],
                "debug": str(e)
            }
        )
