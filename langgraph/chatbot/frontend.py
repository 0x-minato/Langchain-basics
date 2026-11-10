import streamlit as st
from chatbot import chat_bot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# ********************************************* utility functions *****************************************************

def generate_thread_id() -> str:
    thread_id = str(uuid.uuid4())
    return thread_id

def reset_chat():
    new_thread_id = generate_thread_id()
    st.session_state['thread_id'] = new_thread_id
    add_thread(new_thread_id)
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def get_config():
    return {
        'configurable': {'thread_id': st.session_state['thread_id']},
        'metadata': {
            'session_id': st.session_state['thread_id'], 
            'thread_id': st.session_state['thread_id']
        },
        'run_name': 'chat_turn'
    }

def load_conversation(thread_id):
    values = chat_bot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    ).values
    return values.get("messages", [])

# ********************************************* session setup *****************************************************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])

# ********************************************* Sidebar UI *****************************************************

st.sidebar.title("LangGraph bot")
if st.sidebar.button("Start new chat"):
    reset_chat()
st.sidebar.header("My Conversations")

for thread_id in st.session_state['chat_threads'][::-1]:
        messages = load_conversation(thread_id)
        label = messages[0].content if messages else 'New Chat'
    
        if st.sidebar.button(label, key=str(thread_id)):
            st.session_state['thread_id'] = thread_id
            temp_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    role = 'user'
                else: 
                    role = 'assistant'
                temp_messages.append({'role': role, 'content': msg.content})

            st.session_state['message_history'] = temp_messages

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# ********************************************* User flow *****************************************************

user_input = st.chat_input('Type here')

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    status = st.empty()
    status.text('model is thinking...')
    # with st.chat_message('assistant'):
    #     def ai_only_stream():
    #         for message_chunk, metadata in chat_bot.stream(
    #             {'messages': HumanMessage(content=user_input)},
    #             config=get_config(),
    #             stream_mode='messages'
    #         ): 
    #             if isinstance(message_chunk, AIMessage):
    #                 yield message_chunk.content
    #     ai_message = st.write_stream(ai_only_stream())

    result = chat_bot.invoke({'messages': HumanMessage(content=user_input)}, config=get_config())
    status.empty()
    ai_message = result['messages'][-1].content
    with st.chat_message('assistant'):
        st.text(ai_message)

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
              