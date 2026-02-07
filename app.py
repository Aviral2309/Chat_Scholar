
import os, re
from flask import Flask, render_template, request, redirect
from PyPDF2 import PdfReader


from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain


from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
genai.configure(api_key=GOOGLE_API_KEY)

start_greeting = ['hi','hello']
end_greeting = ['bye']
way_greeting = ['who are you?']

# using this folder for staring the uploaded docs. creates the folder at runtime if not present
DATA_DIR = "__data__"  ## whatever data which will be uploading in a frontend, it will store as a local data
os.makedirs(DATA_DIR, exist_ok=True)


app = Flask(__name__)
vectorstore = None
conversation_chain = None
chat_history = []
rubric_text = ""

class HumanMessage:
    def __init__(self, content):
        self.content = content
    
    def __repr__(self):
        return f'HumanMessage(content={self.content})'
    
class AIMessage:
    def __init__(self, content):
        self.content = content
        
    def __repr__(self):
        return f'AIMessage(content={self.content})'

## pdf extraction
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        filename = os.path.join(DATA_DIR, pdf.filename)
        pdf_reader = PdfReader(pdf)
        pdf_txt = ""

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
                pdf_txt += page_text

        with open(filename, "w", encoding="utf-8") as f:
            f.write(pdf_txt)

    return text

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text


def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(text_chunks, embeddings)


def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )


def grade_essay_with_gemini(essay):
    # Load Gemini Pro model
    model = genai.GenerativeModel("gemini-2.5-pro")
    prompt = f"""
You are a strict English essay evaluator.
Please grade the following essay based on the given scoring rubric.
Respond only in English.

[Scoring Rubric]
{rubric_text}

[Essay]
{essay}
"""
    response = model.generate_content(prompt)

    output = response.text if response.text else "Unable to generate grading result"

    return re.sub(r"\n", "<br>", output)

@app.route('/')
def home():
    return render_template('new_home.html')

@app.route('/process', methods=['POST'])
def process_documents():
    global vectorstore, conversation_chain
    pdf_docs = request.files.getlist('pdf_docs')

    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vectorstore)

    return redirect('/chat')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global chat_history
    if request.method == 'POST':
        user_question = request.form['user_question']
        response = conversation_chain({"question": user_question})
        chat_history = response["chat_history"]

    return render_template('new_chat.html', chat_history=chat_history)

@app.route('/pdf_chat')
def pdf_chat():
    return render_template('new_pdf_chat.html')

@app.route('/essay_grading', methods=['GET', 'POST'])
def essay_grading():
    global rubric_text
    result = None
    text = ""

    if request.method == 'POST':
        if request.form.get('essay_rubric'):
            rubric_text = request.form.get('essay_rubric')
            return render_template('new_essay_grading.html')

        if request.files.get('file') and request.files['file'].filename:
            text = extract_text_from_pdf(request.files['file'])
        else:
            text = request.form.get('essay_text')

        result = grade_essay_with_gemini(text)

    return render_template(
        'new_essay_grading.html',
        result=result,
        input_text=text
    )

@app.route('/essay_rubric')
def essay_rubric():
    return render_template('new_essay_rubric.html')

if __name__ == "__main__":
    app.run(debug=True)
