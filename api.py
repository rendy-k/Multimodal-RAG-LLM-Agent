import uvicorn
from fastapi import FastAPI
from validator import ChatQuery
import functions

app = FastAPI()

@app.post("/query/")
def input_question(query_body: ChatQuery):
    response = functions.input_query(query_body)
    return response

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
