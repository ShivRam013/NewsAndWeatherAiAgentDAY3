import streamlit as st
from dotenv import load_dotenv
import os
import requests

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.tools import tool
from tavily import TavilyClient
from rich import print
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage

# =========================
# 🌦️ Weather Tool
# =========================

@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"

    response = requests.get(url)
    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {desc}, {temp}°C"


# =========================
# 📰 News Tool
# =========================

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city: str) -> str:
    """Get latest news"""

    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    results = response.get("results", [])

    if not results:
        return f"No news found for {city}"

    news_list = []

    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")

        news_list.append(f"- {title}\n🔗 {url}\n📝 {snippet[:100]}...")

    return "Latest news:\n\n" + "\n\n".join(news_list)



# =========================
# 🧠 LLM + Agent
# =========================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    max_tokens=200
)

agent = create_agent(
    llm,
    tools=[get_weather, get_news],
    system_prompt="You are a helpful city assistant."
)


# =========================
# 🎨 Streamlit UI
# =========================

st.set_page_config(page_title="City Agent Chatbot", page_icon="🤖")
st.title("NEWS AND WEATHER AI AGENT")

if "chat" not in st.session_state:
    st.session_state.chat = []


# show chat history
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# user input
user_input = st.chat_input("Ask about weather or news...")

if user_input:
    # show user message
    st.chat_message("user").write(user_input)
    st.session_state.chat.append({"role": "user", "content": user_input})

    # agent response
    result = agent.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })

    response = result["messages"][-1].content

    # show bot message
    st.chat_message("assistant").write(response)
    st.session_state.chat.append({"role": "assistant", "content": response})