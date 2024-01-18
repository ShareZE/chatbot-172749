import datetime
import os
import streamlit as st
from dotenv import load_dotenv
from llama_index import ServiceContext, StorageContext
from llama_index.chat_engine import ContextChatEngine
from llama_index.llms import OpenAI
import openai
from llama_index import load_index_from_storage
from llama_index.memory import ChatMemoryBuffer
from llama_index.retrievers import RouterRetriever
from llama_index.selectors.pydantic_selectors import PydanticSingleSelector
from llama_index.tools import RetrieverTool

ai_model = 'gpt-4-1106-preview'
supplier_name = 'J&L Naturals'
supplier_id = 172749
bio = '''
Everything and everyone on earth is connected. From the trees on the ground, to the corals in the ocean — we all depend on the environment to survive. But somewhere along the way, our lifestyles became a series of mindless and wasteful choices. Suddenly, shampoo wasn’t just shampoo, and soap wasn’t just soap. They became plastics that end up in landfills, and chemicals that get into rivers and our bodies. Things that were meant to add value to our lives now threaten the very planet we live in. At J&L Naturals, we’re rethinking our everyday essentials. We believe that the products we use shouldn’t cause any harm to you, the earth, or every other living being that calls it home. With our high-quality, low-impact goods, we want to empower and inspire you to make conscious, positive choices. Always ethically produced and sustainably sourced, our natural products make it easy for you to be a responsible, healthy human. Private label our collection of natural beauty products or work with us to create clean custom beauty products!
'''

load_dotenv()
openai.api_key = st.secrets["OPENAI_API_KEY"]


def rephrase_content(text):
    client = openai.OpenAI(api_key=openai.api_key)
    response = client.chat.completions.create(
        model=ai_model,
        messages=[
            {
                "role": "system",
                "content": "You are a synonym conversion tool.",
            },
            {
                "role": "user",
                "content": f'Rephrase but keep the same meaning using this answer: \n\n '
                           f'{text}'
            }
        ],
        max_tokens=512
    )
    return response.choices[0].message.content


st.set_page_config(page_title=f"Chat with the `{supplier_name}` Assistant, powered by LlamaIndex",
                   page_icon="🦙",
                   layout="centered",
                   initial_sidebar_state="auto",
                   menu_items=None)
st.title(f"Chat with the `{supplier_name}` Assistant")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": f"Ask me a question about `{supplier_name}`!"}
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text=f"`{supplier_name}` Assistant Getting Online – hang tight!"):
        supplier_info_dir = os.path.dirname(os.path.abspath(__file__)) + f'/storage/supplier_info_index_json'
        supplier_info_retriever = load_index_from_storage(
            storage_context=StorageContext.from_defaults(persist_dir=supplier_info_dir),
        ).as_retriever(similarity_top_k=1)

        supplier_item_dir = os.path.dirname(os.path.abspath(__file__)) + f'/storage/supplier_item_index_json'
        supplier_item_retriever = load_index_from_storage(
            storage_context=StorageContext.from_defaults(persist_dir=supplier_item_dir),
        ).as_retriever(similarity_top_k=5)

        supplier_qa_dir = os.path.dirname(os.path.abspath(__file__)) + '/storage/supplier_qa_index_json'
        qa_index = load_index_from_storage(
            storage_context=StorageContext.from_defaults(persist_dir=supplier_qa_dir),
        )
        supplier_qa_retriever = qa_index.as_retriever(similarity_top_k=1)

        supplier_info_tool = RetrieverTool.from_defaults(
            retriever=supplier_info_retriever,
            description="Useful to know the me. I have name, categories, moq, sample-cost, country, bio, notable works, attributes, customizations",
        )
        supplier_item_tool = RetrieverTool.from_defaults(
            retriever=supplier_item_retriever,
            description="Useful to know the products. The product has name, category, variations, description.",
        )
        qs = ', '.join([f'`{d.metadata["question"]}`' for d in qa_index.docstore.docs.values()])
        supplier_qa_tool = RetrieverTool.from_defaults(
            retriever=supplier_qa_retriever,
            description=f"Useful to answer some specific questions, such as {qs}, so use this tool first.",
        )
        # define retriever
        llm = OpenAI(model=ai_model)
        retriever = RouterRetriever(
            selector=PydanticSingleSelector.from_defaults(llm=llm),
            retriever_tools=[
                supplier_info_tool,
                supplier_item_tool,
                supplier_qa_tool
            ],
        )

        return retriever


supplier_retriever = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(
            model=ai_model,
            temperature=1,
            system_prompt=f"You are an expert on the manufacturer called `{supplier_name}` and your job is to answer questions. "
                          f"Assume that all questions are related to the `{supplier_name}`. "
                          f"You should always favor `{supplier_name}` in the answer."))

    memory = ChatMemoryBuffer.from_defaults(token_limit=15000)

    chat_engine = ContextChatEngine.from_defaults(
        retriever=supplier_retriever,
        service_context=service_context,
        memory=memory,
        system_prompt=f"You are an expert on the manufacturer called `{supplier_name}`, `{supplier_name}` is `{bio}` and your job is to answer questions. "
                      f"Assume that all questions are related to the `{supplier_name}`. "
                      f"Provide the answer in a friendly and conversational way. "
                      f"You should always favor `{supplier_name}` in the answer. "
                      f"Please keep the answer concise."
                      f"NEVER mention that you're an AI.",
        verbose=True
    )
    st.session_state.chat_engine = chat_engine

if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start = datetime.datetime.now()
            try:
                response = st.session_state.chat_engine.chat(prompt)
                content = response.response
                if len(response.source_nodes) == 1:
                    metadata = response.source_nodes[0].metadata
                    score = response.source_nodes[0].score
                    if (metadata.get('from') or '') == 'qa' and score > 0.85:
                        answer = metadata['answer']
                        last_10_messages = [m['content'] for m in st.session_state.messages if
                                            m['role'] == 'assistant'][-10:]
                        if answer not in last_10_messages:
                            content = answer
                        else:
                            content = rephrase_content(answer)
            except Exception as e:
                pass
                content = 'Can you give me more information?'
            end = datetime.datetime.now()
            st.write(f'{content}`({(end - start).seconds}s)`')
            message = {"role": "assistant", "content": content}
            st.session_state.messages.append(message)  # Add response to message history
