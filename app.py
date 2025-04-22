import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

# Streamlit interface
st.title("AI Agent ")

# Set OpenAI API Key
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
os.environ["OPENAI_API_KEY"] = openai_api_key
if openai_api_key:
    st.success("API Key Loaded")
else:
    st.warning("Please enter your OpenAI API Key")

# Initialize the LLM
if openai_api_key:
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-0125",
        openai_api_key=openai_api_key
    )
    st.success("LLM initialized successfully!")
    
    # Define the tool (TavilySearchResults)
    tavily_api_key = st.text_input("Enter Tavily API Key", type="password")
    if tavily_api_key:
        tools = [TavilySearchResults(tavily_api_key=tavily_api_key, max_results=1)]
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant. Use the tavily_search_results_json tool for information."),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        # Construct the Tools agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # Input field for user query
        user_input = st.text_input("Ask your question:")

        if user_input:
            response = agent_executor.invoke({"input": user_input})
            st.write(response)
    else:
        st.warning("Please enter your Tavily API Key")
