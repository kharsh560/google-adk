# main.py
import io
import asyncio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse 
import base64
from agent import call_agent_async

app = FastAPI(title="prompt+file processor API")

@app.post("/process")

async def process(prompt: str = Form(...), file: UploadFile = File(...)):
    try:
        # read bytes from UploadFile
        contents = await file.read()  # UploadFile.read() is async
        base64_blob = base64.b64encode(contents).decode("utf-8")
        
        # Call the agent directly (it's already async)
        result_text = await call_agent_async(query=prompt, base64_blob=base64_blob)
        
        return JSONResponse({"result": result_text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
# uvicorn main:app --reload --host 127.0.0.1 --port 8000
