from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import tiktoken

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

# ---------------- TOKEN COUNTER ----------------
def count_tokens(text, model="gemini-3-flash-preview"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# ---------------- MODEL LIMITS ----------------
def get_model_context_limit(model_name: str):
    MODEL_LIMITS = {
        "llama-3.3-70b-versatile": 131072,   # Groq
        "gemini-3-flash-preview": 1048576,   # Gemini
        "gemini-3-pro-preview": 1048576,
    }
    return MODEL_LIMITS.get(model_name, 8192)

SAFE_BUFFER = 2000

# ---------------- TEMPERATURE ----------------
def get_temperature(model_name):
    while True:
        try:
            val = float(input(f"Enter temperature for {model_name}: "))
            if 0 <= val <= 1:
                return val
            else:
                print("Temperature must be between 0 and 1.")
        except ValueError:
            print(f"Invalid input for {model_name}. Please enter a valid number.")

# ---------------- CONTEXT HELPERS ----------------
def build_context(messages):
    return "\n".join([f"{m.type}: {m.content}" for m in messages])

def trim_messages(messages, max_tokens):
    trimmed = messages[:]

    while True:
        text = build_context(trimmed)
        tokens = count_tokens(text)

        if tokens < (max_tokens - SAFE_BUFFER):
            break

        if len(trimmed) <= 1:
            break

        trimmed.pop(0)

    return trimmed

def print_context_usage(name, messages, max_tokens):
    text = build_context(messages)
    used = count_tokens(text)

    print(f"\n🧠 {name} Context Usage:")
    print(f"Used: {used} / {max_tokens}")
    print(f"Remaining: {max_tokens - used}")
    print(f"Usage: {(used/max_tokens)*100:.2f}%")
    print("-" * 100)

# ---------------- MODELS ----------------
model1_temp = get_temperature("Groq")
model2_temp = get_temperature("Gemini")

model1 = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=model1_temp)
model2 = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=model2_temp)

model1_limit = get_model_context_limit(model1.model_name)
model2_limit = get_model_context_limit(model2.model)

print(f"\nGroq context: {model1_limit}")
print(f"Gemini context: {model2_limit}\n")

# ---------------- PROMPT ----------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{query}")
])

chain1 = prompt | model1 | StrOutputParser()
chain2 = prompt | model2 | StrOutputParser()

# ---------------- MEMORY ----------------
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain1_with_memory = RunnableWithMessageHistory(
    chain1,
    get_session_history,
    input_messages_key="query",
    history_messages_key="history",
)

chain2_with_memory = RunnableWithMessageHistory(
    chain2,
    get_session_history,
    input_messages_key="query",
    history_messages_key="history",
)

# ---------------- LOOP ----------------
print("\nType 'exit' to stop\n")

session_id = "user_1"

while True:
    user_query = input("You: ")

    if user_query.lower() == "exit":
        history = store.get(session_id)
        if history:
            print("\n📜 Full Conversation:\n")
            for msg in history.messages:
                print(f"{msg.type.upper()}: {msg.content}\n")
        break

    history = get_session_history(session_id)

    # 🔥 build separate views
    messages_for_model1 = trim_messages(history.messages, model1_limit)
    messages_for_model2 = trim_messages(history.messages, model2_limit)

    # invoke models with separate histories
    response1 = chain1.invoke({
        "query": user_query,
        "history": messages_for_model1
    })

    response2 = chain2.invoke({
        "query": user_query,
        "history": messages_for_model2
    })

    print(f"\nModel 1 ({model1.model_name}): {response1}")
    print("-" * 100)
    print(f"Model 2 ({model2.model}): {response2}")
    print("=" * 100)

    # manually store conversation (important now)
    history.add_user_message(user_query)
    history.add_ai_message(response1)
    history.add_ai_message(response2)

    # usage per model
    print_context_usage("Groq", messages_for_model1, model1_limit)
    print_context_usage("Gemini", messages_for_model2, model2_limit)
