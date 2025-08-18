
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import ConversationalRetrievalChain


def join_text(docs):
    return "\n\n".join([d.page_content for d in docs])


def loan_rag(llm):
    # Load the saved embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="saved_models/all-MiniLM-L6-v2")

    # Load vector db
    load_db = FAISS.load_local(
        'vector_store/product_description/', embeddings, allow_dangerous_deserialization=True
    ) 

    system_prompt = """
    Answer based on the context: {context}.
    If the the context does not provide the answer to the query, admit it. The answer is less than 50 words.
    Answer directly without saying 'based on the context'.
    """

    chat_promp_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

    retriever = load_db.as_retriever(search_kwargs={"k": 5})

    def join_text(docs):
        return "\n\n".join([d.page_content for d in docs])

    rag_chain = (
        {"context": retriever | join_text, "question": RunnablePassthrough()}
        | chat_promp_template
        | llm
        | StrOutputParser()
    )

    return rag_chain
