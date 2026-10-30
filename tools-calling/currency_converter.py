from langchain_core.tools import tool
import requests
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from langchain_core.tools import InjectedToolArg
from typing import Annotated

load_dotenv()

# filled by get_conversion_factor, injected into convert (LLM never sees this arg)
state = {"conversion_rate": None}

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> str:
    """
    Fetch the live conversion factor from base_currency to target_currency.
    Call this first. Then you MUST call convert with the amount.
    """
    url = (
        "https://v6.exchangerate-api.com/v6/"
        f"f098860e0db46fab5320f044/pair/{base_currency}/{target_currency}"
    )
    data = requests.get(url, timeout=20).json()
    state["conversion_rate"] = float(data["conversion_rate"])
    # do not return the number — otherwise the model skips convert and multiplies itself
    return (
        f"Got the {base_currency}->{target_currency} rate. "
        "Now call convert with base_currency_value set to the amount the user asked to convert. "
        "Do not calculate the result yourself."
    )

@tool
def convert(
    base_currency_value: int,
    conversion_rate: Annotated[float, InjectedToolArg],
) -> float:
    """
    Convert an amount using the injected live conversion_rate.
    Pass only base_currency_value (the amount). Call this AFTER get_conversion_factor.
    """
    return base_currency_value * conversion_rate

model = ChatCohere(
    model="command-a-03-2025",
    temperature=0,
)

model_tool = model.bind_tools([get_conversion_factor, convert])

tools_by_name = {
    "get_conversion_factor": get_conversion_factor,
    "convert": convert,
}

messages = [
    SystemMessage(
        "You are a currency assistant. You must use tools. "
        "Never multiply or convert amounts in your own text. "
        "Step 1: call get_conversion_factor. "
        "Step 2: after that tool returns, call convert with base_currency_value. "
        "Only after convert returns may you write the final answer."
    ),
    HumanMessage(
        "What is the conversion rate factor between USD and INR, "
        "and based on that convert 10 USD to INR."
    ),
]

for _ in range(6):
    ai_message = model_tool.invoke(messages)
    messages.append(ai_message)

    if not ai_message.tool_calls:
        print(ai_message.content)
        break

    print(ai_message.tool_calls)

    for call in ai_message.tool_calls:
        name = call["name"]
        if name == "convert":
            if state["conversion_rate"] is None:
                raise RuntimeError("convert called before get_conversion_factor")
            call = {
                **call,
                "args": {**call["args"], "conversion_rate": state["conversion_rate"]},
            }
        tool_message = tools_by_name[name].invoke(call)
        print(tool_message)
        messages.append(tool_message)
else:
    raise RuntimeError("tool loop did not finish")
