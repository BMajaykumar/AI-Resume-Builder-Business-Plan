import os
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain.agents import tool, AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools.tavily_search import TavilySearchResults
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
weather_api_key = os.getenv("WEATHER_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Gemini Model Setup
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    google_api_key=google_api_key
)

# Weather Tool
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city using WeatherAPI."""
    url = f"http://api.weatherapi.com/v1/current.json?key={weather_api_key}&q={city}"
    try:
        response = requests.get(url)
        data = response.json()
        if "current" in data:
            condition = data["current"]["condition"]["text"]
            temp_c = data["current"]["temp_c"]
            return f"The weather in {city} is currently {condition} with {temp_c}°C."
        else:
            return f"Could not fetch weather for {city}."
    except Exception as e:
        return f"Error: {str(e)}"

# Register Weather Tool
weather_tool = Tool.from_function(
    func=get_weather,
    name="get_weather",
    description="Use this tool to get the current weather in a city."
)

# Tavily Search Tool
search_tool = TavilySearchResults(tavily_api_key=tavily_api_key, max_results=3)

# System Prompt for Agent
system_prompt = PromptTemplate.from_template(
    """You are an intelligent travel assistant AI. You help users with two tasks:
1. Provide the current weather in a given city using the weather tool.
2. List top tourist attractions in that city using the search tool.

Use the tools as needed to answer the user's query.
User question: {input}

{agent_scratchpad}"""
)

# Create Tool-Calling Agent
tools = [weather_tool, search_tool]
agent = create_tool_calling_agent(llm, tools, system_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Streamlit UI
st.set_page_config(page_title="Gemini Travel Assistant", page_icon="🌍")
st.title("Gemini Travel Assistant 🌏")
st.write("Enter a city name to get weather and top attractions.")

city = st.text_input("Enter destination city")

if st.button("Find Info") and city:
    with st.spinner("Thinking..."):
        prompt = f"Give me the current weather and top tourist attractions in {city}."
        result = agent_executor.invoke({"input": prompt})
        st.subheader("Travel Assistant AI Response:")
        st.write(result["output"])
elif city:
    st.info("Click the button to get travel information.")
