from agent.state import State

def should_end(state:State):
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "finish"
    else:
        return "Use Tools"