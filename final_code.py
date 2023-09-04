import os
import streamlit as st
import uvicorn
from datetime import datetime
from receipt import receipt_ananlysis
from dotenv import load_dotenv
from dotenv import dotenv_values
import os
load_dotenv()
aws_access_key_id = os.getenv("aws_access_key_id")
aws_secret_access_key = os.getenv("aws_secret_access_key")


# Set OpenAI API key
# from constants import openai_key
# os.environ["OPENAI_API_KEY"] = openai_key
# openai.api_key = os.getenv("OPENAI_API_KEY")

# Download NLTK tokenizer data
# import nltk
# nltk.download('punkt')

def main():

    st.title("Receipt_Extarction")

    # Create "docs" directory if it doesn't exist
    DOCS_FOLDER = "receipt_storage"
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)

    st.sidebar.title("Document Uploader")

    # File uploader in the sidebar
    uploaded_files = st.sidebar.file_uploader("Upload multiple .docx, .txt, and .pdf files", accept_multiple_files=False)

    # Save uploaded files to "docs" folder
    if uploaded_files:
        for file in uploaded_files:
            file_path = os.path.join(DOCS_FOLDER, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        st.sidebar.success("Files saved successfully.")

    # Get the URL of the "docs" folder
    docs_folder_url = os.path.abspath(DOCS_FOLDER)

    # Display the URL to the user
    # st.sidebar.text("URL of 'docs' folder: " + docs_folder_url)

    # Load documents from the "docs" folder
    loader = DirectoryLoader(docs_folder_url)
    documents = loader.load()

    # # Document splitting and summarization
    # text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    # docs_raw_text = [doc.page_content for doc in documents]
    # docs = text_splitter.create_documents(docs_raw_text)

    

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=300)

    # docs1 = loader1.load_and_split()
    docs_raw = loader.load()
    docs_raw_text = [doc.page_content for doc in docs_raw]
    docs = text_splitter.create_documents(docs_raw_text)



    # Initialize OpenAI LLM
    llm = OpenAI(openai_api_key=openai.api_key)

    # # Print sub-chunk tokens length in documents
    # st.text("Printing sub-chunk tokens length in documents:")
    # for sub_chunk in docs:
    #     st.text(llm.get_num_tokens(sub_chunk.page_content))

    # Summarize documents using the LLMChain
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    chunks = chain.run(docs)

    prompt_template_Title = """
    Just use the information provided in the summary of documents to write a suitable one line Title of the Patent Application having summary of invention is: \nSummary:"{text}"\n\n, don't try to make up an answer generic. There are certain specific terms that are not good to use in a patent document. Terms like "invention"(unless it is being used in a header), "consists of", "about", "always", "critical" etc. They are known as "profanity lists".
   
    * The below mentioned examples of Patent Application Title are given for your better understanding.
    * IMPROVED BACKSIDE CONTACT METAL FILL  
    * VOLUME REPLICATION OF STATEFUL-SETS
    * SYSTEM AND METHOHD FOR IDENTIFYING UNPERMITTED DATA IN SOURCE CODE
    * SUPPLYING POWER TO AN ELECTRIC VEHICLE
    * BATTERY PACK
    """

    prompt_Title = PromptTemplate(
        template=prompt_template_Title,
        input_variables=["text"]
    )

    openai_llm = OpenAI(verbose=True,temperature=0.5)
    chain_Title = LLMChain(llm=openai_llm, prompt=prompt_Title)

    response_Title = chain_Title.run(text=chunks)
    # print(response)
    st.title("Title:")
    # st.text(response_Title)
    # Display the summarized data in a text area
    st.text_area("", value=response_Title, height=30)


    if st.button('Regenerate Title'):
        title_text = chain_Title.run(text=chunks)

    # st.text_area(title_text)





    # For Background
    prompt_template_Background = """Write a Background section for a Patent Application based on the provided document summaries. The summary of the invention is: "{text}".

    The Background section should include the "Technical Field" and "Description of the Related Art" subsections and be approximately 110 words long, covering both subsections. Ensure the response is complete.

    In the 'Description of the Related Art' section, focus on explaining the prior art without discussing the actual invention. Address the deficiencies of the prior art and describe the current state of the industry without mentioning the improvements provided by the invention being drafted. This will contextualize the reader and portray the prior art as somewhat inferior compared to the invention without directly discussing the invention itself.

    Example Background Section:

    "Technical Field"
    [0001] The present disclosure relates to fabrication methods and resulting structures for semiconductor devices, specifically focusing on improved backside contact metal fill.

    "Description of the Related Art"
    [0002] As semiconductor footprints continue to shrink, concerns arise about unfavorable contact between components. Three-dimensional transistors, such as FinFETs and GAA FETs, offer improved performance but present challenges in terms of alignment margins and reduced fin widths. Backside power rails have issues with shorting, leakage, routing resistance, and packing density.

    [0003] The Direct Backside Contact (DBC) process enables CMOS scaling by providing wiring alternatives on the wafer's backside. It eliminates concerns about narrow space formations and high aspect ratio backside power rails. The DBC process involves creating deep trenches for backside source/drain contacts.
    
    Another Example Background Section:

    Technical Field
    [0001]	The present disclosure generally relates to volume replication of stateful-sets using application behavior, and more particularly, to decoupling volumes and reducing locking pressure based on the dynamic behavior and access pattern of workloads.
    Description of the Related Art
    [0002]	Container management systems are developed for managing container lifecycle (Create, Read, Update, and Delete (CRUD)) in a cluster-wide system. As an example, once a container creation request is received, a scheduler decides the host where requested containers will run, and then an agent in the host launches the container.
    [0003]	However, the container management systems are not without deficiency. For example, response latency and CPU utilization may increase when the number of client threads increases due to locking pressure on volumes.

    
    """ 



    prompt_Background = PromptTemplate(
    template=prompt_template_Background,
    input_variables=["text"]
    )

    # openai_llm = OpenAI(verbose=True,temperature=0.5)
    chain_Background = LLMChain(llm=openai_llm, prompt=prompt_Background)

    response_Background = chain_Background.run(text=chunks)
    # print(response)
    st.title("Background:")
    # st.text(response_Background)

    # # Display the summarized data in a text area
    st.text_area("", value=response_Background, height=500)

    

    # # Perform other operations with the summarized data as needed
    # st.text("Summarized data:")
    # for chunk in chunks:
    #     st.text(chunk.page_content)

if __name__ == "__main__":
    main()





# # import os
# # import openai
# # import streamlit as st
# # from langchain.text_splitter import CharacterTextSplitter
# # from langchain.document_loaders import UnstructuredFileLoader
# # # from pdfminer3 import PdfReader

# # from constants import openai_key

# # os.environ["OPENAI_API_KEY"] = openai_key
# # openai.api_key = os.getenv("OPENAI_API_KEY")

# # from langchain.docstore.document import Document

# # import nltk
# # nltk.download('punkt')

# # from langchain.document_loaders import DirectoryLoader

# # loader1 = DirectoryLoader('/content/gdrive/MyDrive/Tanveer_Kofi_provided_data_latest/1')
# # documents1 = loader1.load()


# # from langchain.chains.summarize import load_summarize_chain

# # text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=300)

# # # docs1 = loader1.load_and_split()
# # docs1_raw = loader1.load()
# # docs1_raw_text = [doc.page_content for doc in docs1_raw]
# # docs1 = text_splitter.create_documents(docs1_raw_text)

# # print("Printing sub chunk tokens length in document1")
# # for sub_chunk in docs1:
# #   print(llm.get_num_tokens(sub_chunk.page_content))


# # chain = load_summarize_chain(llm, chain_type="map_reduce")
# # chunks1 = chain.run(docs1)

# # from langchain import OpenAI
# # llm = OpenAI(openai_api_key=openai.api_key)

# # import openai
# # import langchain
# # from langchain.llms import OpenAI 
# # from langchain.prompts import PromptTemplate
# # from langchain.chains import RetrievalQA
# # from langchain.document_loaders import DirectoryLoader

# # prompt_template = """
# # Just use the information provided in the summary of documents \nSummary:"{text}"\n\n, don't try to make up an answer generic. There are certain specific terms that are not good to use in a patent document. Terms like "invention"(unless it is being used in a header), "consists of", "about", "always", "critical" etc. They are known as "profanity lists".

# # The below mentioned examples of Patent Application Title are given for your better understanding.
# # * IMPROVED BACKSIDE CONTACT METAL FILL  
# # * VOLUME REPLICATION OF STATEFUL-SETS
# # * SYSTEM AND METHOHD FOR IDENTIFYING UNPERMITTED DATA IN SOURCE CODE
# # * SUPPLYING POWER TO AN ELECTRIC VEHICLE
# # * BATTERY PACK
# # """

# # prompt = PromptTemplate(
# #     template=prompt_template,
# #     input_variables=["text"]
# # )


# # from langchain.llms import OpenAI
# # from langchain.chains import LLMChain

# # openai_llm = OpenAI(verbose=True,temperature=0.5)
# # chain = LLMChain(llm=openai_llm, prompt=prompt)

# # response = chain.run(text=chunks1)
# # print(response)


# def load_document():
#     """
#     Load a document from a file upload.

#     Args:
#         uploaded_files (list): A list of file uploads.

#     Returns:
#         list: A list of text chunks.
#     """
#     # with open("C:\\Users\\ClickNET\\OneDrive\\Desktop\\Streamlit_Langchain\\docs\\Input Invention Disclosure.DOCX") as f:
#     #     document_text=f.readlines()
#     # print("______________________")
#     # print(document_text)
#     raw_text = ''
#     for uploaded_file in os.listdir('C:/Users/ClickNET/OneDrive/Desktop/Streamlit_Langchain/docs/'):
#         loader = UnstructuredFileLoader(uploaded_file)
#         doc = loader.load()


#     text_splitter = CharacterTextSplitter(
#         separator="\n",  # line break
#         chunk_size=1000,
#         chunk_overlap=200,  # Striding over the text
#         length_function=len,
#     )
#     texts = text_splitter.split_text(doc)
#     return texts


# def apply_prompts(pieces, prompts):
#     """
#     Apply prompts to a list of text chunks.

#     Args:
#         pieces (list): A list of text chunks.
#         prompts (list): A list of prompts.

#     Returns:
#         list: A list of responses.
#     """

#     responses = []
#     for piece in pieces:
#         response = openai.Completion.create(
#             engine="text-davinci-003",
#             prompt=f"What is the main topic of this document?",
#             max_tokens=100,
#             temperature=0.7,
#             top_p=0.9,
#         )
#         responses.append(response.choices[0].text)
#     return responses


# def main():
#     """
#     Main function.
#     """

#     st.title("WEB-APP PATENT APPLICATION")

    
#     def save_file(uploaded_file):
#         folder_path = "C:\\Users\\ClickNET\\OneDrive\\Desktop\\Streamlit_Langchain\\docs"
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)
#         file_path = os.path.join(folder_path, uploaded_file.name)
#         st.write(file_path)
#         with open(file_path, "rb") as f:
#             text=f.readlines()
#         st.write(text)
        
#         st.success("File saved successfully!")

    
#     uploaded_file = st.sidebar.file_uploader("Upload an Invention Disclosure Document", type=["pdf", "docx", "txt"])
#     # st.write(uploaded_file)

#     if uploaded_file is not None:
#         save_file(uploaded_file)


#     # Apply the prompts
#     prompts = [f"Write a suitable one-line title for this invention document"]
#     responses = apply_prompts(load_document(), prompts)
#     print("responses",len(responses))
#     st.write(f"**Title of Invention:** {responses[0]}")


# if __name__ == "__main__":
#     main()



# # import os
# # import openai
# # import streamlit as st
# # from langchain.text_splitter import CharacterTextSplitter
# # from langchain.document_loaders import UnstructuredFileLoader

# # from constants import openai_key

# # os.environ["OPENAI_API_KEY"] = openai_key
# # openai.api_key = os.getenv("OPENAI_API_KEY")
# # # openai = "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"
# # # OPENAI_API_KEY = "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"

# # def load_document(uploaded_files):
# #     # text_splitter = CharacterTextSplitter(chunk_size=1026, chunk_overlap=100)
# #     # pieces = text_splitter.split_documents(filename)

# #     raw_text = ''
# #     # Loop through each uploaded file
# #     for uploaded_file in uploaded_files:
      
# #       loader = UnstructuredFileLoader(uploaded_file)
# #       doc_read = loader.load()
      
# #     #   pdf_reader = PdfReader(uploaded_file)# Read the PDF
# #       for i, page in enumerate(doc_read.pages):        # Loop through each page in the PDF
    
# #             # Extract the text from the page
# #             text = page.extract_text()
        
# #             # If there is text, add it to the raw text
# #             if text:
# #               raw_text += text
              
# #     # Split text into smaller chucks to index them
# #     text_splitter = CharacterTextSplitter(
# #                     separator="\n", # line break
# #                     chunk_size = 1000,
# #                 # Striding over the text
# #                 chunk_overlap = 200,  
# #                 length_function=len,
# #     )
    
# #     texts = text_splitter.split_text(raw_text)
# #     # document_splitter = document_splitters.UnstructuredWordDocumentSplitter()
# #     # pieces = document_splitter.split(filename)
# #     return pieces

# # def apply_prompts(pieces, prompts):
# #     responses = []
# #     for piece in pieces:
# #         response = openai.Completion.create(
# #             engine="text-davinci-003",
# #             prompt=f"What is the main topic of this document?",
# #             max_tokens=100,
# #             temperature=0.7,
# #             top_p=0.9,
# #         )
# #         responses.append(response.choices[0].text)
# #     return responses

# # st.title("WEB-APP PATENT APPLICATION")




# # st.sidebar.markdown("**Upload an Invention Disclosure Document**")
# # document_file = st.sidebar.file_uploader("Upload a document")






# # if document_file is not None:
# #     # Load the document
# #     document = load_document(document_file)

# #     # # Split text into smaller chucks to index them
# #     # text_splitter = CharacterTextSplitter(
# #     #                 separator="\n", # line break
# #     #                 chunk_size = 1000,
# #     #             # Striding over the text
# #     #             chunk_overlap = 200,  
# #     #             length_function=len,
# #     # )
    
# #     # texts = text_splitter.split_text(document)
    
# #     # # Download embeddings from OPENAI
# #     # embeddings = OpenAIEmbeddings() # Default model "text-embedding-ada-002"
    
# #     # Create a FAISS vector store with all the documents and their embeddings
# #     docsearch = FAISS.from_texts(document, embeddings)
    
# #     # Load the question answering chain and stuff it with the documents
# #     chain = load_qa_chain(OpenAI(), chain_type="stuff", verbose=True) 

# #     query = st.text_input("Ask a question or give an instruction")

# #     if query:
# #       # Perform a similarity search to find the 6 most similar documents "chunks of text" in the corpus of documents in the vector store
# #       docs = docsearch.similarity_search(query, k=6)
      
# #       # Run the question answering chain on the 6 most similar documents based on the user's query
# #       answer = chain.run(input_documents=docs, question=query)
      
# #       # Print the answer and display the 6 most similar "chunks of text" vectors 
# #       st.write(answer, docs[0:6])

# #     # Apply the prompts
# #     prompts = [f"Write a suitable one-line title for this invention document"]
# #     responses = apply_prompts(document, prompts)











# # if uploaded_files:

# #     raw_text = ''
# #     # Loop through each uploaded file
# #     for uploaded_file in uploaded_files:
# #       pdf_reader = PdfReader(uploaded_file)# Read the PDF
# #       for i, page in enumerate(pdf_reader.pages):        # Loop through each page in the PDF
    
# #             # Extract the text from the page
# #             text = page.extract_text()
        
# #             # If there is text, add it to the raw text
# #             if text:
# #               raw_text += text
              
# #     # Split text into smaller chucks to index them
# #     text_splitter = CharacterTextSplitter(
# #                     separator="\n", # line break
# #                     chunk_size = 1000,
# #                 # Striding over the text
# #                 chunk_overlap = 200,  
# #                 length_function=len,
# #     )
    
# #     texts = text_splitter.split_text(raw_text)
    
# #     # Download embeddings from OPENAI
# #     embeddings = OpenAIEmbeddings() # Default model "text-embedding-ada-002"
    
# #     # Create a FAISS vector store with all the documents and their embeddings
# #     docsearch = FAISS.from_texts(texts, embeddings)
    
# #     # Load the question answering chain and stuff it with the documents
# #     chain = load_qa_chain(OpenAI(), chain_type="stuff", verbose=True) 

# #     query = st.text_input("Ask a question or give an instruction")
    
# #     if query:
# #       # Perform a similarity search to find the 6 most similar documents "chunks of text" in the corpus of documents in the vector store
# #       docs = docsearch.similarity_search(query, k=6)
      
# #       # Run the question answering chain on the 6 most similar documents based on the user's query
# #       answer = chain.run(input_documents=docs, question=query)
      
# #       # Print the answer and display the 6 most similar "chunks of text" vectors 
# #       st.write(answer, docs[0:6])















# # # st.write(data)

# # # if uploaded_file is not None:
# # #     # Load the document
# # #     document = load_document(data)
    

# # #     # Prompt the user to write a one-line title
# # #     title = st.text_input("Write a one-line title for your invention disclosure document")

# # #     # Apply the prompts
# # #     prompts = [f"What is the main topic of this document?"]
# # #     responses = apply_prompts(document, prompts)

# # #     # Display the responses
# # #     st.write(f"**Title:** {title}")
# # #     st.write(f"**Main topic:** {responses[0]}")

# # # Load the document
# # document = load_document(data)


# # # Prompt the user to write a one-line title
# # # title = st.text_input("Write a one-line title for your invention disclosure document")

# # # Apply the prompts
# # prompts = [f"Write a suitable one line title for this invention document"]
# # responses = apply_prompts(document, prompts)

# # # Display the responses
# # # st.write(f"**Title:** {title}")
# # st.write(f"**Title of Invention:** {responses[0]}")






# # import os
# # import openai
# # from PyPDF2 import PdfReader
# # from langchain.embeddings.openai import OpenAIEmbeddings
# # from langchain.text_splitter import CharacterTextSplitter
# # from langchain.vectorstores import FAISS
# # from langchain.llms import OpenAI
# # from langchain.chains.question_answering import load_qa_chain
# # import streamlit as st
# # import tiktoken
# # from constants import openai_key

# # # Load environment variables from .env file

# # os.environ["OPENAI_API_KEY"] = openai_key
# # # Streamlit Code for UI - Upload PDF(s)
# # st.title('ChatPDF :microphone:')

# # uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
# # if uploaded_files:
# #     raw_text = ''
# #     # Loop through each uploaded file
# #     for uploaded_file in uploaded_files:
# #       pdf_reader = PdfReader(uploaded_file)# Read the PDF
# #       for i, page in enumerate(pdf_reader.pages):        # Loop through each page in the PDF
    
# #             # Extract the text from the page
# #             text = page.extract_text()
        
# #             # If there is text, add it to the raw text
# #             if text:
# #               raw_text += text
              
# #     # Split text into smaller chucks to index them
# #     text_splitter = CharacterTextSplitter(
# #                     separator="\n", # line break
# #                     chunk_size = 1000,
# #                 # Striding over the text
# #                 chunk_overlap = 200,  
# #                 length_function=len,
# #     )
    
# #     texts = text_splitter.split_text(raw_text)
    
# #     # Download embeddings from OPENAI
# #     embeddings = OpenAIEmbeddings() # Default model "text-embedding-ada-002"
    
# #     # Create a FAISS vector store with all the documents and their embeddings
# #     docsearch = FAISS.from_texts(texts, embeddings)
    
# #     # Load the question answering chain and stuff it with the documents
# #     chain = load_qa_chain(OpenAI(), chain_type="stuff", verbose=True) 

# #     query = st.text_input("Ask a question or give an instruction")
    
# #     if query:
# #       # Perform a similarity search to find the 6 most similar documents "chunks of text" in the corpus of documents in the vector store
# #       docs = docsearch.similarity_search(query, k=6)
      
# #       # Run the question answering chain on the 6 most similar documents based on the user's query
# #       answer = chain.run(input_documents=docs, question=query)
      
# #       # Print the answer and display the 6 most similar "chunks of text" vectors 
# #       st.write(answer, docs[0:6])
