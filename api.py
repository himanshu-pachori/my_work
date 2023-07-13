from typing import Dict
from pydantic import BaseModel
from fastapi import FastAPI
import os
import uvicorn
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

saved_tokenizer = AutoTokenizer.from_pretrained('./models/t5-base-finetuned_TextToSql')
saved_model = AutoModelForSeq2SeqLM.from_pretrained('./models/t5-base-finetuned_TextToSql')

app = FastAPI()

print("loading tokenizer + model")

saved_tokenizer = AutoTokenizer.from_pretrained('./models/t5-base-finetuned_TextToSql')
saved_model = AutoModelForSeq2SeqLM.from_pretrained('./models/t5-base-finetuned_TextToSql')

print("loaded tokenizer + model")

class Request(BaseModel):
    text: str

class Response(BaseModel):
    Query: str

@app.get('/')
async def index():
    return {"message" : "Convert Text to Sql"}


@app.post("/predict", response_model=Response)
async def predict(request: Request):
    conversion_text_sample = f'text to sql: {request.text}'

    #output = sorted(CLF(request.text)[0], key=lambda x: x['score'], reverse=True)  # use our pipeline and sort results

    input_ids = saved_tokenizer(conversion_text_sample, return_tensors='pt').input_ids

    outputs = saved_model.generate(input_ids, max_length= 512)
    query = saved_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return Response(
        Query = query
    )


#if __name__ == '__main__':
#    uvicorn.run("app:app", host="127.0.0.1", port=5050, reload= True)