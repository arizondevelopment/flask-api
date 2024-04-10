import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
from flask_pymongo import PyMongo
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from bson import ObjectId
from bson.json_util import dumps
import os, sys
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import openai
from bs4 import BeautifulSoup
import requests
from pathlib import Path
import nltk
import camelot
from PyPDF2 import PdfFileReader
from camelot.core import TableList
import os
import argparse
import subprocess
import logging
import tabula
import json

nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import tiktoken
model = "gpt-4-turbo-preview"
enc = tiktoken.encoding_for_model(model)
# Load environment variables from a .env file
load_dotenv()

# Set your OpenAI API key from the environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the Flask app and enable CORS
app = Flask(__name__)
CORS(app, origins="*")
app.config['CORS_HEADERS'] = 'Content-Type'

# Configure MongoDB
app.config["MONGO_URI"] = os.getenv("MONGO_URI")
mongo = PyMongo(app)
db = mongo.db


def parse_json_garbage(s):
    s = s[next(idx for idx, c in enumerate(s) if c in "{["):]
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        return json.loads(s[:e.pos])

# Function to process a PDF file and extract text
def process_pdf(file_path, pages):
    try:
        with open(file_path, 'rb') as f:
            pdf_reader = PdfReader(f)
            text = ""
            for page_num in range(len(pdf_reader.pages)):                
                if pages == "all" or str(page_num) in pages:
                    text += pdf_reader.pages[page_num].extract_text()
        return text
    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)}")
        return None

def total_pages(pdf):
    with open(pdf, 'rb') as file:
        pdf_object = PdfFileReader(file)
        pages = ','.join([str(i) for i in range(pdf_object.getNumPages())])
    return pages


def extract_tables(pdf, pattern):
    try:
        cmd = f"pdfgrep -Pn '{pattern}' {pdf} | awk -F\":\" '$0~\":\"{{print $1}}' | tr '\n' ','"
        print(cmd)
        logging.info(cmd)
        pages = subprocess.check_output(cmd, shell=True).decode("utf-8")
        logging.info(f'count of pages {pages}')
        if not pages:
            logging.warning(f"No matching pages found in {pdf}")
            return

        tabula.convert_into(pdf, f"{os.path.splitext(pdf)[0]}.csv", output_format="csv", pages="39")
        # jsonoutput = tabula.read_pdf(pdf, output_format="json", pages="39")
        # print(jsonoutput)
        logging.info(f"Processed {pdf}")
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        logging.error(f"Error processing {pdf}: {str(e)}")
# Function to process a PDF file and store it in the 'uploads' folder
def process_pdf_and_store(file):
    try:
        uploads_folder = os.path.join(os.getcwd(), "uploads")
        if not os.path.exists(uploads_folder):
            os.makedirs(uploads_folder)

        filename = secure_filename(file.filename)
        file_path = os.path.join(uploads_folder, filename)
        file.save(file_path)

        text = process_pdf(file_path, "all")
        if text is not None:
            return filename
        else:
            os.remove(file_path)
            return None
    except Exception as e:
        logging.error(f"Error processing PDF and storing: {str(e)}")
        return None



def tokenizer_length(string: str) -> int:
    """Returns the number of tokens in a text string."""
    return len(enc.encode(string))


def split_text_by_sentences(text, token_limit):
    """Splits a text into segments of complete sentences, each with a number of tokens up to token_limit."""
    sentences = sent_tokenize(text)
    current_count = 0
    sentence_buffer = []
    segments = []

    for sentence in sentences:
        # Estimate the token length of the sentence
        sentence_length = tokenizer_length(sentence)

        if current_count + sentence_length > token_limit:
            if sentence_buffer:
                segments.append(' '.join(sentence_buffer))
                sentence_buffer = [sentence]
                current_count = sentence_length
            else:
                # Handle the case where a single sentence exceeds the token_limit
                segments.append(sentence)
                current_count = 0
        else:
            sentence_buffer.append(sentence)
            current_count += sentence_length

    # Add the last segment if there's any
    if sentence_buffer:
        segments.append(' '.join(sentence_buffer))

    return segments


def extract_json(filename, user_query):
    text = ""
    pdf_path = os.path.join("uploads", filename)
    # cmd = f"pdfgrep -Pn '^(?s:(?=.*Revenue)|(?=.*Income)|(?=.*EBITDA)|(?=.*Annual)|(?=.*Q3))|(?=.*Growth)|(?=.*guidance)' {pdf_path} | awk -F\":\" '$0~\":\"{{print $1}}' | tr '\n' ','"
    # print(cmd)
    # logging.info(cmd)
    # pages = subprocess.check_output(cmd, shell=True).decode("utf-8")
    # logging.info(f'count of pages {pages}')
    # if not pages:
    #    logging.warning(f"No matching pages found in {pdf_path}")
    #    return
    # processed_text = process_pdf(pdf_path, pages)
    processed_text = process_pdf(pdf_path, "all")
    if processed_text is not None:
        text += processed_text
    parts = split_text_by_sentences(text, token_limit=30000)
    print("parts")
    print(len(parts))
    print(type(parts))
    full_completion = {
            "name": "NA",
            "annual_revenue": "NA",
            "EBITA": "NA",
            "annual_growth_percentage": "NA",
            "EBITA_growth_percentage": "NA",
            "guidance_next_year": "NA"
            }

    for index, segment in enumerate(parts):
        try:
            print("inside for")
            openai.api_key = os.environ["OPENAI_API_KEY"]
            print("chat number")
            print(index)
            completion = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a highly experienced Business Analyst and Financial Expert with a rich history of over 30 years in the field. Your expertise is firmly grounded in data-driven insights and comprehensive analysis. When presented with unsorted data, your primary objective is to meticulously filter out any extraneous or irrelevant components, including elements containing symbols like # and $. Furthermore, you excel at identifying and eliminating any HTML or XML tags and syntax within the data, streamlining it into a refined and meaningful form."},
                    {"role": "user", "content": segment},
                    {"role": "assistant", "content": f"Question: {user_query}\nAnswer:"},
                ]
            )
            print(completion.choices[0].message.content.strip() )
            current_result = parse_json_garbage(completion.choices[0].message.content.strip() )
            print(current_result)
            full_completion = {
                "name": current_result['name'] if current_result['name'] != 'NA' else full_completion['name'],
                "annual_revenue": current_result['annual_revenue'] if current_result['annual_revenue'] != 'NA' else full_completion['annual_revenue'],
                "EBITA": current_result['EBITA'] if current_result['EBITA'] != 'NA' else full_completion['EBITA'],
                "annual_growth_percentage": current_result['annual_growth_percentage'] if current_result['annual_growth_percentage'] != 'NA' else full_completion['annual_growth_percentage'],
                "EBITA_growth_percentage": current_result['EBITA_growth_percentage'] if current_result['EBITA_growth_percentage'] != 'NA' else full_completion['EBITA_growth_percentage'],
                "guidance_next_year": current_result['guidance_next_year'] if current_result['guidance_next_year'] != 'NA' else full_completion['guidance_next_year']
                }
            print(full_completion)
        except Exception as e:
            logging.error(f"Error processing chat request for project: {str(e)}")
            return jsonify({"error": f"Scraping error: {str(e)}"}), 500    
    return full_completion;



def split_text_by_tokens(text, token_limit):
    """Splits a text into segments, each with a number of tokens up to token_limit."""
    words = text.split()
    current_count = 0
    word_buffer = []
    segments = []

    for word in words:
        # Add a space for all but the first word in the buffer
        test_text = ' '.join(word_buffer + [word]) if word_buffer else word
        word_length = tokenizer_length(test_text)

        if word_length > token_limit:
            # If a single word exceeds the token_limit, it's added to its own segment
            segments.append(word)
            word_buffer.clear()
            continue

        if current_count + word_length > token_limit:
            segments.append(' '.join(word_buffer))
            word_buffer = [word]
            current_count = tokenizer_length(word)
        else:
            word_buffer.append(word)
            current_count = word_length

    # Add the last segment if there's any
    if word_buffer:
        segments.append(' '.join(word_buffer))

    return segments


# Route for the root endpoint
@app.route("/")
@cross_origin()
def helloWorld():
    return "Hello, API WORLD!"

@app.route('/scrape-and-query', methods=['POST'])
@cross_origin()
def scrape_and_query():
    data = request.get_json()
    urls = data.get('urls', [])
    user_query = data.get('query')

    if not urls or not user_query:
        return jsonify({"error": "URLs or query not provided"}), 400

    try:
        scraped_data = []

        for url in urls:
            # Scrape each website with a timeout of 60 seconds
            response = requests.get(url, timeout=60)
            response.raise_for_status()  # Check for HTTP errors

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                scraped_text = ' '.join([p.get_text() for p in soup.find_all('p')])
                scraped_data.append(scraped_text)

        # Join scraped data from all URLs into a single text
        all_scraped_data = ' '.join(scraped_data)

        # Query OpenAI's Davinci Chat Model
        openai.api_key = os.environ["OPENAI_API_KEY"]
        ai_response = openai.ChatCompletion.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a highly experienced Business Analyst and Financial Expert with a rich history of over 30 years in the field. Your expertise is firmly grounded in data-driven insights and comprehensive analysis. When presented with unsorted data, your primary objective is to meticulously filter out any extraneous or irrelevant components, including elements containing symbols like # and $. Furthermore, you excel at identifying and eliminating any HTML or XML tags and syntax within the data, streamlining it into a refined and meaningful form."},
                {"role": "user", "content": all_scraped_data},
                {"role": "assistant", "content": f"Question: {user_query}\nAnswer:"},
            ],
        )
        answer = ai_response['choices'][0]['message']['content']

        return jsonify({"response": answer})

    except requests.exceptions.RequestException as req_err:
        return jsonify({"error": f"Request error: {str(req_err)}"}), 500

    except Exception as e:
        return jsonify({"error": f"Scraping error: {str(e)}"}), 500




@app.route('/scrape-and-query-pdf/<project_id>', methods=['POST'])
@cross_origin()
def scrape_and_query_pdf(project_id):
    print("calling scrape_and_query_pdf")
    data = request.get_json()
    urls = data.get('urls', [])
    user_query = data.get('query')
    project = db.projects.find_one({"_id": ObjectId(project_id)})
    if not project:
        return jsonify({"error": f"Project with ID '{project_id}' not found"}), 404
    urls = project.get('filenames', [])

    if not urls or not user_query:
        return jsonify({"error": "URLs or query not provided"}), 400
    try:
        mainReport = extract_json(urls[0], user_query)
        compReport = extract_json(urls[1], user_query)



        db.projects.update_one(
        {"_id": ObjectId(project_id)},
        {"$set": {"report": {
            "main": mainReport,
            "comp": compReport
        }}}
        )
        return jsonify({"response": {
            "main": mainReport,
            "comp": compReport
        } })

    except requests.exceptions.RequestException as req_err:
        return jsonify({"error": f"Request error: {str(req_err)}"}), 500

    except Exception as e:
        return jsonify({"error": f"Scraping error: {str(e)}"}), 500


# Route to process and upload a file
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        filename = process_pdf_and_store(file)

        if filename:
            file_url = f"{request.url_root}uploads/{filename}"
            return jsonify({"file_url": file_url, "filename": filename}), 201
        else:
            return jsonify({"error": "Error processing PDF and storing"}), 500

    except Exception as e:
        logging.error(f"Error uploading file: {str(e)}")
        return jsonify({"error": f"Error uploading file: {str(e)}"}), 500

# Route to get all projects or add a new project
@app.route('/projects', methods=['GET', 'POST'])
def projects():
    if request.method == 'GET':
        projects = db.projects.find()
        projects = [{"_id": str(project["_id"]), **project} for project in projects]
        return dumps({"projects": projects}), 200

    elif request.method == 'POST':
        try:
            data = request.get_json()

            if 'name' not in data or 'description' not in data or 'filenames' not in data:
                return jsonify({"error": "Incomplete project information"}), 400




            name = data['name']
            description = data['description']
            filenames = data['filenames']
            jsonfilenames = []

            tables = []
            for filename in filenames:
                pdf_path = os.path.join("uploads", filename)
                print(pdf_path)
                extract_tables(pdf_path, '^(?s:(?=.*Revenue)|(?=.*Income))')
                jsonfilenames.append(filename.replace(".pdf", ".json"))

            project_data = {
                "name": name,
                "description": description,
                "filenames": filenames,
                "jsonfilenames": jsonfilenames
            }

            db.projects.insert_one(project_data)

            return jsonify({"message": "Project created successfully"}), 201
        except Exception as e:
            return jsonify({"error": f"project adding  error: {str(e)}"}), 500
# Route to get a project by ID
@app.route('/projects/<project_id>', methods=['GET'])
def get_project_by_id(project_id):
    try:
        if not ObjectId.is_valid(project_id):
            return jsonify({"error": "Invalid project ID"}), 400

        project = db.projects.find_one({"_id": ObjectId(project_id)})


        if not project:
            return jsonify({"error": f"Project with ID '{project_id}' not found"}), 404

        project["_id"] = str(project["_id"])

        return jsonify({"project": project}), 200

    except Exception as e:
        logging.error(f"Error retrieving project by ID: {str(e)}")
        return jsonify({"error": f"Error retrieving project by ID: {str(e)}"}), 500



# Route to retrieve uploaded files
@app.route('/uploads/<filename>', methods=['GET'])
def get_uploaded_file(filename):
    return send_from_directory(os.path.join(os.getcwd(), "uploads"), filename)

# Function to get or create the vector store for text embeddings
def get_or_create_vector_store(chunks, store_name):
    embeddings_file_path = f"{store_name}.pkl"

    if os.path.exists(embeddings_file_path):
        with open(embeddings_file_path, "rb") as f:
            vector_store = pickle.load(f)
    else:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        with open(embeddings_file_path, "wb") as f:
            pickle.dump(vector_store, f)

    return vector_store

# Chat API route
@app.route('/chat/<project_id>', methods=['POST'])
def chat(project_id):
    try:
        data = request.get_json()
        if 'query' not in data:
            return jsonify({"error": "Query not provided"}), 400

        query = data['query']

        project = db.projects.find_one({"_id": ObjectId(project_id)})
        if not project:
            return jsonify({"error": f"Project with ID '{project_id}' not found"}), 404

        filenames = project.get('filenames', [])
        text = ""
        for filename in filenames:
            pdf_path = os.path.join("uploads", filename)
            processed_text = process_pdf(pdf_path, "all")
            if processed_text is not None:
                text += processed_text

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        vector_store = get_or_create_vector_store(chunks, project_id)

        docs = vector_store.similarity_search(query=query, k=3)
        llm = OpenAI(temperature=0.7, model="gpt-3.5-turbo-instruct")
        chain = load_qa_chain(llm=llm, chain_type="stuff")

        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)

        chat_history = project.get('chat_history', [])
        chat_history.append({"user": query, "bot": response})

        db.projects.update_one(
            {"_id": ObjectId(project_id)},
            {"$set": {"chat_history": chat_history}}
        )

        return jsonify({"response": response})

    except Exception as e:
        logging.error(f"Error processing chat request for project '{project_id}': {str(e)}")
        return jsonify({"error": f"Error processing chat request: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
