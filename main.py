import uvicorn
from datetime import datetime
from typing import List
from fastapi import FastAPI, Query, Request
from fastapi import File, UploadFile
# from fastapi.encoders import jsonable_encodera
from receipt import receipt_ananlysis
from dotenv import load_dotenv
from dotenv import dotenv_values
import os
load_dotenv()
aws_access_key_id = os.getenv("aws_access_key_id")
aws_secret_access_key = os.getenv("aws_secret_access_key")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/UploadImage")
async def upload(file: UploadFile = File(...)):
    print(file)
    file_location=f"receipt_storage/{file.filename}"
    with open(file_location,"wb+") as file_object:
        file_object.write(file.file.read())
    return {"info": f"Successfully uploaded {file.filename} saved at '{file_location}'"}


@app.post("/ReceiptAnalysis")
async def ReceiptExpense(filename: str):
    print(filename)
    file = f"uploaded_files/{filename}"+str(".pdf")
    out = receipt_ananlysis(filename)
    json_output={}
    for i in range(len(out['ExpenseDocuments'][0]["SummaryFields"])):
        complete_response=out['ExpenseDocuments'][0]["SummaryFields"][i]
        json_output[complete_response['Type']["Text"]]=complete_response['ValueDetection']["Text"]
    for i in range(len(out['ExpenseDocuments'][0]["LineItemGroups"][0]["LineItems"][0]["LineItemExpenseFields"])):
        complete_response=out['ExpenseDocuments'][0]["LineItemGroups"][0]["LineItems"][0]["LineItemExpenseFields"][i]
        json_output[complete_response['Type']["Text"]]=complete_response['ValueDetection']["Text"]
    return {"data":json_output}

if __name__ == "__main__":
   uvicorn.run("main:app", port=8501, reload=True)