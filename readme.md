# ArXiv Research with Artificial Intelligence using IBM WatsonX

Today, we are going to build an interesting application that allows you to search files in **ArXiv** using **WatsonX**.

## Introduction

In the world of scientific research, finding relevant information from a vast pool of academic papers can be a daunting task. Traditional search engines often fall short in effectively retrieving the most pertinent articles, hindering progress in finding potential cures and treatments for critical health issues. However, with the advent of AI-powered technologies like WatsonX.ai and Streamlit, researchers now have a powerful tool at their disposal to navigate the wealth of knowledge stored in ArXiv.

In this blog, we will explore how to build an application that utilizes these cutting-edge technologies to answer scientific questions.

![demo](./assets/images/posts/readme/demo.gif)

The high-level structure of the program is as follows:

1. Question Analysis: Analyze your question using the Artificial Intelligence of WatsonX
2. Searching on **ArXiv**: Search for relevant papers on ArXiv
3. Download multiple papers and extract their text content.
4. Text Chunking: Divide the extracted text into smaller chunks that can be processed effectively.
5. Language Model: The application utilizes a language model to generate vector representations (embeddings) of the text chunks.
6. Similarity Matching: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.
7. Response Generation: The selected chunks are passed to the language model, which generates a response based on the relevant contents.

## Step 1: Environment Creation

There are several ways to create an environment in Python. In this tutorial, we will show two options.

1. Conda method:

First, you need to install Anaconda from this [link](https://www.anaconda.com/products/individual). Install it in the location **C:\\Anaconda3** and then check if your terminal recognizes **conda** by typing the following command:

```
C:\\conda --version
conda 23.1.0
```

The environment we will use is Python 3.10.11. You can create an environment called **watsonx** (or any other name you prefer) by running the following command:

```
conda create -n watsonx python==3.10.11
```

After creating the environment, activate it:

```
conda activate watsonx
```

Next, install the necessary packages by running the following command:

```
conda install ipykernel notebook
```

2. Python native method:

First, install Python 3.10.11 from [here](https://www.python.org/downloads/). Then, create a virtual environment by running the following command:

```
python -m venv .venv
```

You will notice a new directory in your current working directory with the same name as your virtual environment. Activate the virtual environment:

```
.venv\\Scripts\\activate.bat
```

Upgrade pip:

```
python -m pip install --upgrade pip
```

Install the notebook package:

```
pip install ipykernel notebook
```

## Step 2: Setup Libraries

Once we have our running environment, we need to install additional libraries. Install the necessary libraries by running the following command:

```
pip install streamlit python-dotenv PyPDF2 arxiv langchain htmlTemplates ibm_watson_machine_learning requests pandas
```

## Step 3: Getting API from IBM Cloud

To obtain an **API key from IBM Cloud**, follow these steps:

1. Sign in to your IBM Cloud account at https://cloud.ibm.com.
2. In the IBM Cloud dashboard, click on your account name in the top right corner.
3. From the dropdown menu, select "Manage" to go to the Account settings.
4. In the left-hand menu, click on "IBM Cloud API keys" under the "Access (IAM)" section.
5. On the "API keys" page, click on the "Create an IBM Cloud API key" button.
6. Provide a name for your API key and an optional description.
7. Select the appropriate access policies for your API key if needed.
8. Click on the "Create" button to generate the API key.
9. Once the API key is created, you will see a dialog box displaying the API key value. Make sure to copy and save this key as it will not be shown again.

Please note that the steps above are based on the current IBM Cloud interface, and the steps may vary slightly depending on any updates or changes made to the IBM Cloud dashboard. If you encounter any difficulties or if the steps do not match your IBM Cloud interface, I recommend referring to the IBM Cloud documentation or contacting IBM support for further assistance.

To obtain the **Project ID for IBM Watsonx**, you will need to have access to the IBM Watson Machine Learning (WML) service. Here are the steps to retrieve the Project ID:

1. Log in to the IBM Cloud Console (https://cloud.ibm.com) using your IBM Cloud credentials.
2. Navigate to the Watson Machine Learning service.
3. Click on the service instance associated with your Watsonx project.
4. In the left-hand menu, click on "Service credentials".
5. Under the "Credentials" tab, you will find a list of service credentials associated with your Watsonx project. Click on the name of the service credential that you want to use.
6. In the JSON object, you will find the "project_id" field. The value of this field is your Project ID.

Add the API key to the `.env` file in the project directory.

```
API_KEY=your_api_key
PROJECT_ID=your_projec_id
```

If you have a high-end NVIDIA GPU card, you can install the pytorch capability with CUDA:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

## Step 4:  Creation of app.py

Create a file `app.py`   with the following code:

```python
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from langchain.vectorstores import FAISS
from langchain.embeddings import TensorflowHubEmbeddings
import requests
import os
import tempfile
import pandas as pd
parameters = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 500,
    GenParams.MIN_NEW_TOKENS: 0,
    GenParams.STOP_SEQUENCES: ["\n"],
    GenParams.REPETITION_PENALTY:1
    }


load_dotenv()
project_id = os.getenv("PROJECT_ID", None)
credentials = {
        "url": "https://us-south.ml.cloud.ibm.com",
        "apikey": os.getenv("API_KEY", None)
        }    
#this cell should never fail, and will produce no output
import requests

def getBearer(apikey):
    form = {'apikey': apikey, 'grant_type': "urn:ibm:params:oauth:grant-type:apikey"}
    print("About to create bearer")
#    print(form)
    response = requests.post("https://iam.cloud.ibm.com/oidc/token", data = form)
    if response.status_code != 200:
        print("Bad response code retrieving token")
        raise Exception("Failed to get token, invalid status")
    json = response.json()
    if not json:
        print("Invalid/no JSON retrieving token")
        raise Exception("Failed to get token, invalid response")
    print("Bearer retrieved")
    return json.get("access_token")

credentials["token"] = getBearer(credentials["apikey"])
from ibm_watson_machine_learning.foundation_models import Model
model_id = ModelTypes.LLAMA_2_70B_CHAT

# Initialize the Watsonx foundation model
llama_model = Model(
    model_id=model_id, 
    params=parameters, 
    credentials=credentials,
    project_id=project_id)

# Function to get text from PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        text += " ".join(page.extract_text() for page in pdf_reader.pages)
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store
def get_vectorstore(text_chunks):
    url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
    embeddings  = TensorflowHubEmbeddings(model_url=url)   
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create a conversation chain
def get_conversation_chain(vectorstore):

    llm=llama_model.to_langchain()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def call_model_flan(question):
    
    parameters = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 50,
    GenParams.MIN_NEW_TOKENS: 1,
    #GenParams.STOP_SEQUENCES: ["\n"],
    GenParams.STOP_SEQUENCES: ["<|endoftext|>"],
    GenParams.REPETITION_PENALTY:1,
    
    }    
    
    # Initialize the Watsonx foundation model
    llm_model= Model(
        model_id=ModelTypes['FLAN_T5_XXL'], 
        params=parameters, 
        credentials=credentials,
        project_id=project_id)
    prompt = f"Considering the following question, generate 3 keywords are most significant to use when searching in the Arxiv API ,provide your response as a Python list: {question}. "
    result=llm_model.generate(prompt)['results'][0]['generated_text']

    # Convert string to a list of individual words
    word_list = result.split(', ')    
    
    return word_list


def download_pdf(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(response.content)

def download_pdf_files(url_list):
    temp_dir = tempfile.gettempdir()  # Get the temporary directory path
    downloaded_files = []  # List to store downloaded file paths
    for i, url in enumerate(url_list):
        filename = os.path.join(temp_dir, f'file_{i+1}.pdf')  # Set the absolute path in the temporary directory
        download_pdf(url, filename)
        downloaded_files.append(filename)  # Append the file name to the list with the path
        print(f'Downloaded: {filename}')

    return downloaded_files  # Return the list of downloaded file names

def delete_files_in_temp():
    temp_dir = tempfile.gettempdir()  # Get the temporary directory path
    for file in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")
            
def arxiv_search(topic):
    import arxiv
    print("Searching on Arxiv: '{}' ".format(topic))
    # combinations of single topics
    titles = list()
    authors = list()
    summary = list()
    pdf_url = list()
    search = arxiv.Search(
    query = topic,
    max_results = 5,
    sort_by = arxiv.SortCriterion.Relevance
    #SubmittedDate #TODO Include it
    )
    print('Fetching items for token: {}'.format(topic))  
    titles = [result.title for result in arxiv.Client().results(search)]
    pdf_url = [result.pdf_url for result in arxiv.Client().results(search)]
    url_list =pdf_url
    downloaded_files = download_pdf_files(url_list)
    return downloaded_files ,titles

# Function to handle user input and display responses
def handle_user_input(user_question, titles=None):
    parameters = {"instruction": "Answer the following question using only information from the article. If there is no good answer in the article, say I don't know"}
    prompt={"question": user_question}
    response = st.session_state.conversation(prompt)
    #st.write(response)
    
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


# Main function
def main():
    st.set_page_config(page_title="Chat with your Documents", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    st.header("Chat with ArXiv Documents :books:")
    user_question = st.text_input("Ask questions to ArXiv or upload your documents:") 
    
    if st.button("Search") and user_question:      
        with st.spinner("Analyzing query"):
            original_list=call_model_flan(user_question)
            unique_list = list(set(original_list))
            topic = ' '.join(unique_list)  # full topic creation
        with st.spinner("Searching in ArXiv: "+topic):
            downloaded_files , titles =arxiv_search(topic)
        with st.spinner("Vectorizing results"):
            # Get PDF text and split into chunks
            raw_text = get_pdf_text(downloaded_files)
            text_chunks = get_text_chunks(raw_text)
            # Create vector store and conversation chain
            vectorstore = get_vectorstore(text_chunks)
            #st.write('Vectorization completed')
            st.write("Documents loaded")
            st.session_state.conversation = get_conversation_chain(vectorstore) 
            if titles is not None:     
                # Using list comprehension with enumeration to create a new list of strings
                enumerated_strings = [f"{index + 1}. {value}" for index, value in enumerate(titles)]
                # Combining the enumerated strings into a single string
                combined_string = ', \n '.join(enumerated_strings)  # Separate each enumerated string by a new line
                st.write(bot_template.replace("{{MSG}}", "On ArXiv I found the following relevant papers: "+combined_string), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if not pdf_docs:
            st.write('You can add your document')
        else:     
            if st.button("Process"):
                with st.spinner("Processing"):
                    # Get PDF text and split into chunks
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    # Create vector store and conversation chain
                    vectorstore = get_vectorstore(text_chunks)
                    st.write("Document loaded")
                    titles=None
                    
                    st.session_state.conversation = get_conversation_chain(vectorstore)                    
    if user_question and st.session_state.conversation is not None:
            handle_user_input(user_question)
        
            
if __name__ == '__main__':
    main()

```

and we add the following CSS file `htmlTemplates.py`

```python
css = '''
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
}

.chat-message.user {
    background-color: #0072CE; /* IBM Watsonx blue color */
}

.chat-message.bot {
    background-color: #F3F3F3; /* Light gray color */
}

.chat-message .avatar {
    width: 20%;
}

.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}

.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #000; /* Black color */
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/DDw07m6/robots.jpg" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/4J4n4Df/user.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

```



## Step 3: Running your program

To use the ArXiv Chatter App, follow these steps:

1. Ensure that you have installed the required dependencies and added the API key to the `.env` file.
2. Run the `main.py` file using the Streamlit CLI. Execute the following command:
```
streamlit run app.py
```
3. The application will launch in your default web browser, displaying the user interface.
4. Load multiple PDF documents into the app by following the provided instructions.
5. Ask questions in natural language about the loaded PDFs using the chat interface.

## Conclusion:

By harnessing the power of AI, specifically WatsonX.ai and Streamlit, we have created an innovative application that revolutionizes the way researchers search in ArXiv. This technology empowers scientists to find solutions to critical health problems efficiently, potentially leading to groundbreaking discoveries and advancements in medical research. With AI as our ally, we can pave the way for a healthier future.

## Troubleshooting

You can get a list of existing Conda environments using the command below:

### Delete an Environment in Conda

```
conda env list
```

Before you delete an environment in Conda, you should first deactivate it. You can do that using this command:

```
conda deactivate
```

Once you've deactivated the environment, you will be switched back to the `base` environment. To delete an environment, run the command below:

```
conda remove --name ENV_NAME --all
```

Faiss issues:

If you encounter the following error:

```
INFO:faiss.loader:Loading faiss with AVX2 support.
INFO:faiss.loader:Could not load library with AVX2 support due to:
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
INFO:faiss.loader:Loading faiss.
INFO:faiss.loader:Successfully loaded faiss.
```

Using Command Prompt (cmd):

1. Open Command Prompt as an administrator.
2. Navigate to the directory where you want to create the symbolic link using the `cd` command. For example, if you want to create the link in your user folder, you can use:

   ```
   cd your_python_path/site-packages/faiss
   ```

   You can retrieve your Python path by typing `conda info`.

3. Create the symbolic link using the `mklink` command as follows:

   ```
   mklink swigfaiss_avx2.py swigfaiss.py
   ```

   This command creates a symbolic link named `swigfaiss_avx2.py` that points to `swigfaiss.py`.

Using Linux:

```
cd your_python_path/site-packages/faiss
ln -s swigfaiss.py swigfaiss_avx2.py
```

## Contributing

This repository is intended for educational purposes.

## License

The ArXiv Chat App is released under the [MIT License](https://opensource.org/licenses/MIT).