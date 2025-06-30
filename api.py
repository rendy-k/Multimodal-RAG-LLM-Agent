import uvicorn
from fastapi import FastAPI
from features.validator import ChatQuery
import features.ai_agent as ai_agent

app = FastAPI()

@app.post("/query/")
def input_question(query_body: ChatQuery):
    response = ai_agent.input_query(query_body)
    return response

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
