from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",max_tokens=200)
parser = StrOutputParser()
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that answers questions in easy way."),
        ("human", "Question: {question}")
    
    ]
)   
chain = prompt |model | parser
while True:
    text = input("You: Enter your question ") 
    result = chain.invoke({"question": text})
    print("AI :",result)
