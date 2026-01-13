import os
import time
import uuid
import tempfile
import shutil
import traceback
import pathlib
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from supabase import create_client, Client
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber

# -----------------------------
# 1️⃣ Load environment variables
# -----------------------------
current_dir = pathlib.Path(__file__).parent
env_path = current_dir / ".env"
load_dotenv(env_path)

GENAI_API_KEY = os.getenv("GENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not GENAI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("Missing environment variables. Check your .env file.")

# -----------------------------
# 2️⃣ Initialize clients
# -----------------------------
client = genai.Client(api_key= os.getenv("GENAI_API_KEY"))
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    api_key=os.getenv("GENAI_API_KEY")
)

# -----------------------------
# 3️⃣ FastAPI setup
# -----------------------------
app = FastAPI(title="PDF AI Query API", version="1.0.1")

origins = [
    "https://pdf-ai-frontend-six.vercel.app",  # your frontend URL
    "http://localhost:3000",  # for local development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # allow these origins
    allow_credentials=True,
    allow_methods=["*"],     # allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],     # allow all headers
)

# -----------------------------
# 4️⃣ Request models
# -----------------------------
class QueryRequest(BaseModel):
    question: str
    session_id: str
    match_count: int = 5

class EndSessionRequest(BaseModel):
    session_id: str

# -----------------------------
# 5️⃣ Helper: Extract PDF text
# -----------------------------
def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            if len(pdf.pages) == 0:
                raise HTTPException(status_code=400, detail="PDF is empty or corrupted")
            
            for page_num, page in enumerate(pdf.pages, 1):
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
    except pdfplumber.PDFSyntaxError:
        raise HTTPException(status_code=400, detail="Invalid or corrupted PDF")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract PDF text: {str(e)}")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="No readable text found in PDF")
    
    return text

# -----------------------------
# 6️⃣ Helper: Delete session data
# -----------------------------
def delete_session_data(session_id: str):
    try:
        # Force returning count instead of representation
        result = supabase.table("pdf_vectors").delete().eq("session_id", session_id).execute()
        # Check affected rows
        deleted_count = result.count if hasattr(result, "count") else len(result.data) if result.data else 0
        print(f"Deleted {deleted_count} records for session: {session_id}")
        return deleted_count
    except Exception as e:
        print(f"Failed to delete session data for {session_id}: {e}")
        return 0


# -----------------------------
# 7️⃣ Helper: Process PDF and store embeddings
# -----------------------------
def process_pdf_and_store(session_id: str, pdf_text: str):
    try:
        # Clear old data for session first
        delete_session_data(session_id)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(pdf_text)
        if not chunks:
            raise HTTPException(status_code=400, detail="No text could be extracted from PDF")
        
        print(f"Generated {len(chunks)} text chunks")

        vector_data = embeddings.embed_documents(chunks)
        print(f"Generated {len(vector_data)} embeddings")

        records = []
        for i, chunk in enumerate(chunks):
            records.append({
                "id": str(uuid.uuid4()),
                "session_id": session_id,
                "content": chunk,
                "embedding": vector_data[i]
            })

        # Insert in batches
        batch_size = 20
        total_inserted = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            try:
                supabase.table("pdf_vectors").insert(batch).execute()
                total_inserted += len(batch)
            except Exception as batch_error:
                print(f"Batch insert error: {batch_error}")
                for record in batch:
                    try:
                        supabase.table("pdf_vectors").insert(record).execute()
                        total_inserted += 1
                    except Exception as single_error:
                        print(f"Failed single insert: {single_error}")

        print(f"Stored {total_inserted} chunks for session {session_id}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing PDF: {traceback.format_exc()}")
        # Cleanup partial data
        delete_session_data(session_id)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

# -----------------------------
# 8️⃣ Upload PDF
# -----------------------------
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), session_id: str = None):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds 50MB limit")

    # Use existing session_id if provided, otherwise generate a new one
    session_id = session_id or str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp(prefix="pdf_upload_")
    temp_path = os.path.join(temp_dir, f"{session_id}.pdf")

    try:
        with open(temp_path, "wb") as f:
            f.write(content)
        pdf_text = extract_text_from_pdf(temp_path)
        process_pdf_and_store(session_id, pdf_text)
        return {
            "session_id": session_id,
            "message": "PDF uploaded and processed successfully",
            "filename": file.filename,
            "size": len(content),
            "text_length": len(pdf_text),
            "status": "success"
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        shutil.rmtree(temp_dir, ignore_errors=True)


# -----------------------------
# 9️⃣ Ask PDF question
# -----------------------------

@app.post("/ask")
def ask_pdf(request: QueryRequest):
    if not request.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # -----------------------------
        # 1️⃣ Generate question embedding
        # -----------------------------
        try:
            question_embedding = embeddings.embed_query(request.question)
            print(f"[DEBUG] Question embedding generated, length: {len(question_embedding)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {e}")

        # -----------------------------
        # 2️⃣ Fetch similar chunks from Supabase
        # -----------------------------
        try:
            response = supabase.rpc("find_similar_chunks", {
                "query_embedding": question_embedding,
                "match_limit": request.match_count,
                "target_session": request.session_id
            }).execute()
            if not response.data:
                return {
                    "question": request.question,
                    "answer": "No relevant content found in your PDF.",
                    "session_id": request.session_id,
                    "status": "no_matches"
                }
            print(f"[DEBUG] Found {len(response.data)} similar chunks")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase RPC failed: {e}")

        # -----------------------------
        # 3️⃣ Prepare prompt for Gemini
        # -----------------------------
        context_chunks = [
            f"[Chunk {i+1}, Similarity: {r.get('similarity_score',0):.3f}]\n{r.get('chunk_content','')}"
            for i, r in enumerate(response.data)
        ]
        context = "\n\n".join(context_chunks)

        prompt = f"""You are a helpful PDF assistant that answers questions based ONLY on the provided PDF content.

PDF CONTENT:
{context}

USER QUESTION:
{request.question}

INSTRUCTIONS:
1. If the question is unrelated to the PDF (like greetings or small talk), respond politely in a natural conversational way.
2. Always provide a meaningful response.
3. If answer not in PDF, say "The PDF does not contain information about this" politely.
4. Clear, structured, plain text answer.
5. Be concise but thorough and wellformatted like heading paragraph.
6. Answer should be properly formatted into sentences, points, and paragraphs, not just plain text.
7. Remove characters like *, # in the content.
ANSWER:"""

        # -----------------------------
        # 4️⃣ Try generating answer with models
        # -----------------------------
        model_priority = [
            "models/gemini-2.5-flash",
            "models/gemini-2.5-flash-lite",
            "models/gemini-2.5-flash-tts",
            "models/gemini-3-flash",
            "models/gemini-2.0-flash",
        ]

        last_error = None
        for model_name in model_priority:
            try:
                answer = client.models.generate_content(model=model_name, contents=prompt)
                response_text = getattr(answer, "text", "")
                if response_text:
                    print(f"[DEBUG] Answer generated with model: {model_name}")
                    return {
                        "question": request.question,
                        "answer": response_text.strip(),
                        "session_id": request.session_id,
                        "model_used": model_name,
                        "status": "success"
                    }
            except Exception as e:
                last_error = e
                error_str = str(e)
                print(f"[DEBUG] Model {model_name} failed: {error_str}")

                # If quota exhausted, stop trying other models (likely share same quota)
                if "RESOURCE_EXHAUSTED" in error_str:
                    return {
                        "question": request.question,
                        "answer": "Sorry, the AI service quota has been exceeded. Please try again later.",
                        "session_id": request.session_id,
                        "model_used": model_name,
                        "status": "quota_exceeded"
                    }
                # Otherwise, try next model
                continue

        # -----------------------------
        # 5️⃣ If all models fail
        # -----------------------------
        return {
            "question": request.question,
            "answer": "Sorry, the AI could not generate an answer at this time.",
            "session_id": request.session_id,
            "status": "failed",
            "error": str(last_error) if last_error else None
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# -----------------------------
# 🔟 End session
# -----------------------------
@app.post("/end_session")
def end_session(request: EndSessionRequest):
    if not request.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required")
    
    deleted_count = delete_session_data(request.session_id)
    return {
        "message": "Session ended successfully",
        "session_id": request.session_id,
        "deleted_count": deleted_count,
        "status": "success"
    }

# -----------------------------
# 🔟 Health check
# -----------------------------
@app.get("/health")
def health_check():
    try:
        supabase_response = supabase.table("pdf_vectors").select("id", count="exact").limit(1).execute()
        test_embedding = embeddings.embed_query("test")
        dummy_vector = [0.0] * len(test_embedding)
        supabase.rpc("find_similar_chunks", {
            "query_embedding": dummy_vector,
            "match_limit": 1,
            "target_session": "test"
        }).execute()
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# -----------------------------
# Root
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "PDF AI Query API", "version": "1.0.1"}

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
