import operator
import os
import asyncio
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY", "")
openrouter_llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-3.5-turbo",
)

class ZipOutput(BaseModel):
    zip_code: str = Field(description="Extracted ZIP code or NONE")
    reply: str = Field(description="The AI's response message")

class ChatState(TypedDict):
    messages: list
    zip_code: str | None
    attempts: int


async def extract_zip_llm(state: ChatState):
    print("DEBUG: extract_zip_llm invoked with state:", state)
    if not state["messages"]:
        return {
            "messages": state["messages"],
            "zip_code": None,
            "attempts": state["attempts"]
        }
    
    #read the latest user input
    user_msg = state["messages"][-1].content
    attempts = state["attempts"]
    zip_code = None
    try:
        zip_parser = PydanticOutputParser(pydantic_object=ZipOutput)

        prompt = f"""
            You are an AI agent extracting ZIP codes.

            The user's message is:
            "{user_msg}"

            Instructions:
            1. Extract ONLY the ZIP code, it can be any ZIP code of any country.
            2. If you cannot find a ZIP code, reply exactly with: NONE and ask user for zip again politely
            3. Do NOT guess. Only return a ZIP code if you are 100% sure.
            4. Output must be ONLY the ZIP code (e.g., 94016 ,90210-1234) or NONE.
            5. Return your answer in JSON and ONLY JSON:
            {{ "zip_code": "<value>", "reply": "<message to user>" }}
            6. Output MUST be only JSON in this exact schema:
            {zip_parser.get_format_instructions()}
            """
        
        result = await openrouter_llm.ainvoke([HumanMessage(content=prompt)])

        raw_text = result.content.strip()
        parsed = zip_parser.parse(raw_text)

        # if parsed.zip_code.isdigit() and len(parsed.zip_code) == 5:
            #     extracted_zip = parsed.zip_code
            # else:
            #     extracted_zip = "NONE"

        extracted_zip = parsed.zip_code
    except Exception as e:
        print("DEBUG: Exception during LLM parsing:", e)
        if user_msg:
            for word in user_msg.split():
                if word.isdigit() and len(word) == 5:
                    extracted_zip = word
                else:
                    parsed = ZipOutput(zip_code="NONE", reply="I couldn't find a valid ZIP code. Could you please provide it again?")
                    extracted_zip = parsed.zip_code

    if extracted_zip != "NONE":
        zip_code = extracted_zip
        return {
            "messages": state["messages"] + [AIMessage(content=f"Thanks! ZIP {extracted_zip} received.")],
            "zip_code": zip_code,
            "attempts": attempts + 1
        }

    if extracted_zip == "NONE" and attempts <5:
        return {
            "messages": state["messages"] + [AIMessage(content=parsed.reply),],
            "zip_code": None,
            "attempts": attempts + 1
        }
    
def router(state: ChatState):
    print("DEBUG: router invoked with state:", state)
    if state["zip_code"]:
        return "save_zip"
    if state["attempts"] >= 5:
        return "end"
    return "ask_zip"

async def save_zip(state: ChatState):
    print("DEBUG: save_zip invoked with state:", state)
    zip_code = state["zip_code"]
    return {}
   

async def end_node(state: ChatState):
    print("DEBUG: end_node invoked with state:", state)
    return {
        "messages": state["messages"] + [AIMessage(content="No ZIP code provided after multiple attempts. Goodbye!")],
        "zip_code": state["zip_code"],
        "attempts": state["attempts"]
    }

graph = StateGraph(ChatState)
graph.add_node("extract_zip_llm", extract_zip_llm)

graph.add_node("save_zip", save_zip)
graph.add_node("end", end_node)

graph.set_entry_point("extract_zip_llm")

graph.add_conditional_edges(
    "extract_zip_llm",
    router,
    {
        "ask_zip": END,
        "save_zip": "save_zip",
        "end": "end"
    }
)

graph.add_edge("save_zip", END)
graph.add_edge("end", END)

workflow = graph.compile()

async def run_chat():
    state = {"messages": [], "zip_code": None, "attempts": 0}
    
    print("BOT: Please provide your 5-digit ZIP code.")
    
    while state["attempts"] < 5 and not state["zip_code"]:
        user_msg = input("YOU: ")
        
        #add user message to state
        state["messages"] = state["messages"] + [HumanMessage(content=user_msg)]

        print(f"DEBUG: Current state before invoking workflow: {state}")
        
        #invoke workflow with updated state
        result = await workflow.ainvoke(state)
        print(f"DEBUG: Result from workflow: {result}")
        
        #update state with result
        state = result
        print(f"DEBUG: Updated state after invoking workflow: {state}")

        if state["messages"] and len(state["messages"]) > 0:
            last_msg = state["messages"][-1]
            if isinstance(last_msg, AIMessage):
                print("BOT:", last_msg.content)

if __name__ == "__main__":
    asyncio.run(run_chat())