import os
import getpass
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

os.environ['OPENAI_API_KEY'] = ''
#activeloop_token = getpass.getpass("eyJhbGciOiJIUzUxMiIsImlhdCI6MTY5MDI1NzEwMywiZXhwIjoxNjkzMDIxODU5fQ.eyJpZCI6InJlbnpvYmFsY2F6YXIxIn0.XRdPTt5cBLNzQ86gSmXWf3uQWi1V4VssiDnus6vUGmhAUqes-OW6RFJfjVfVIAY4rwa25hKQQFxigp2mXaKBUg")
embeddings = OpenAIEmbeddings(disallowed_special=())


root_dir = "./topo_cell_seg"
docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        try:
            loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
            docs.extend(loader.load_and_split())
        except Exception as e:
            pass

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

username = "renzobalcazar1"  
db = DeepLake(
    dataset_path=f"hub://renzobalcazar1/topo_cell_seg",
    embedding_function=embeddings,
)
db.add_documents(texts)

db = DeepLake(
    dataset_path="hub://renzobalcazar1/topo_cell_seg",
    read_only=True,
    embedding_function=embeddings,
)

retriever = db.as_retriever()
retriever.search_kwargs["distance_metric"] = "cos"
retriever.search_kwargs["fetch_k"] = 100
retriever.search_kwargs["maximal_marginal_relevance"] = True
retriever.search_kwargs["k"] = 10

model = ChatOpenAI(model_name="gpt-3.5-turbo")  
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

questions = [
    "how to run this code?",
]
chat_history = []

for question in questions:
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")