import os
import shutil
import uvicorn
from fastapi import FastAPI,File,UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from helpers import generate_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def index():
    return {"ESRGAN": 'This Model Generate Super Resolution Image'}

@app.post('/uploadfile/')
async def create_upload_file(file:UploadFile = File(...)):
    with open(os.path.join(f"images/input/", file.filename),'wb+') as buffer:
        shutil.copyfileobj(file.file,buffer)
    input_file_path = f"images/input/{file.filename}"
    output_file_name = generate_image(input_file_path)
    output_file_path = f"images/output/{output_file_name}"
    return FileResponse(output_file_path)

if __name__=="__main__":
    uvicorn.run(app,host="127.0.0.1",port=8000)