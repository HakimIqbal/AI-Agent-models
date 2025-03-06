from fastapi import FastAPI
from pydantic import BaseModel
from rag_agent import qa_chain

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_agent(query: Query):
    response = qa_chain.run(query.question)
    return {"answer": response}
