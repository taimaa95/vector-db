from langchain.text_splitter import RecursiveCharacterTextSplitter

def make_text_splitter():
    # Keep your previous chunking (tweak sizes only if they MATCH your old ones)
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )

def split_documents(docs):
    splitter = make_text_splitter()
    return splitter.split_documents(docs)
