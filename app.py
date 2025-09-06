
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
import os
import requests
from dotenv import load_dotenv
import google.generativeai as genai
import fitz # PyMuPDF
import numpy as np
import faiss
import pickle # Import pickle for saving/loading Python objects
import threading # Import threading for background tasks

load_dotenv() # Load environment variables from .env file

app = Flask(__name__)

# Google AI configuration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Generative Model
print("Available Models:")
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)
gen_model = genai.GenerativeModel('gemini-2.5-pro')

# Twilio credentials
ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")

client = Client(ACCOUNT_SID, AUTH_TOKEN)

PDFS_FOLDER = 'pdfs'

RAG_DATA_FOLDER = 'rag_data'
FAISS_INDEX_PATH = os.path.join(RAG_DATA_FOLDER, 'faiss_index.bin')
TEXT_CHUNK_STORE_PATH = os.path.join(RAG_DATA_FOLDER, 'text_chunk_store.pkl')

def load_rag_data():
    global faiss_index, text_chunk_store, embedding_dimension
    os.makedirs(RAG_DATA_FOLDER, exist_ok=True)
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(TEXT_CHUNK_STORE_PATH):
        print("Loading existing RAG data...")
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(TEXT_CHUNK_STORE_PATH, 'rb') as f:
            text_chunk_store = pickle.load(f)
        print(f"Loaded {faiss_index.ntotal} vectors and {len(text_chunk_store)} text chunks.")
    else:
        print("No existing RAG data found. Initializing new Faiss index and text store.")
        # Ensure embedding_dimension is defined before initializing Faiss index
        embedding_dimension = 768 # Default if not loaded
        faiss_index = faiss.IndexFlatL2(embedding_dimension)
        text_chunk_store = []

def save_rag_data():
    print("Saving RAG data...")
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(TEXT_CHUNK_STORE_PATH, 'wb') as f:
        pickle.dump(text_chunk_store, f)
    print("RAG data saved successfully.")

# Faiss Index and Text Storage
# Initialize an empty Faiss index and a list to store corresponding text chunks
# The dimension of the embeddings from text-embedding-004 is 768
embedding_dimension = 768
faiss_index = faiss.IndexFlatL2(embedding_dimension)
text_chunk_store = []

# Load RAG data on startup
load_rag_data()

def retrieve_relevant_chunks(query, k=3):
    if faiss_index.ntotal == 0:
        return []
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return []
    query_embedding_np = np.array([query_embedding]).astype('float32')
    distances, indices = faiss_index.search(query_embedding_np, k)
    
    # Retrieve the actual text chunks
    relevant_chunks = [text_chunk_store[idx] for idx in indices[0] if idx < len(text_chunk_store)]
    return relevant_chunks

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return None

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - chunk_overlap)
    return chunks

def get_embedding(text):
    try:
        return genai.embed_content(model="models/text-embedding-004", content=text)['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

if not os.path.exists(PDFS_FOLDER):
    os.makedirs(PDFS_FOLDER)

# Function to send messages directly via Twilio Client
def send_whatsapp_message(to, body):
    try:
        client.messages.create(
            from_='whatsapp:+14155238886', # Replace with your Twilio WhatsApp number
            to=to,
            body=body
        )
        print(f"Sent WhatsApp message to {to}: {body}")
    except Exception as e:
        print(f"Error sending WhatsApp message to {to}: {e}")

def process_whatsapp_message_in_background(from_number, incoming_msg, num_media, media_url=None, media_content_type=None):
    if num_media > 0 and media_url and media_content_type == 'application/pdf':
        try:
            filename = os.path.join(PDFS_FOLDER, f'{os.urandom(16).hex()}.pdf')
            response = requests.get(media_url, auth=(ACCOUNT_SID, AUTH_TOKEN))
            
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
                send_whatsapp_message(from_number, "Thank you for sending the PDF! It has been successfully stored and is now being processed to answer your questions. This may take a moment.")
                
                extracted_text = extract_text_from_pdf(filename)
                if extracted_text:
                    print(f"""Successfully extracted text from {filename}:\n{extracted_text[:500]}...""")
                    text_chunks = chunk_text(extracted_text)
                    print(f"Created {len(text_chunks)} chunks from the PDF.")
                    if text_chunks:
                        print(f"First chunk: {text_chunks[0][:200]}...")
                        embeddings = [get_embedding(chunk) for chunk in text_chunks if get_embedding(chunk) is not None]
                        print(f"Generated {len(embeddings)} embeddings.")
                        if embeddings:
                            embeddings_np = np.array(embeddings).astype('float32')
                            faiss_index.add(embeddings_np)
                            text_chunk_store.extend(text_chunks)
                            print(f"Added {len(embeddings)} embeddings to Faiss index. Total vectors: {faiss_index.ntotal}")
                            save_rag_data()
                            send_whatsapp_message(from_number, "Your PDF has been processed and saved! You can now ask questions about its content.")
                        else:
                            send_whatsapp_message(from_number, "Successfully stored the PDF, but failed to generate embeddings from its text.")
                else:
                    send_whatsapp_message(from_number, "Successfully stored the PDF, but failed to extract text from it.")
            else:
                print(f"Failed to download PDF from {media_url}. Status Code: {response.status_code}, Response: {response.text}")
                send_whatsapp_message(from_number, "Failed to download the PDF. Please try again.")
        except Exception as e:
            print(f"Error in background PDF processing: {e}")
            send_whatsapp_message(from_number, "An unexpected error occurred while processing your PDF. Please try again.")

    elif incoming_msg and num_media == 0:
        if faiss_index.ntotal > 0:
            print(f"User query: {incoming_msg}")
            relevant_chunks = retrieve_relevant_chunks(incoming_msg)
            if relevant_chunks:
                print(f"Retrieved {len(relevant_chunks)} relevant chunks:")
                for j, chunk in enumerate(relevant_chunks):
                    print(f"Chunk {j+1}: {chunk[:200]}...")
                
                context = "\n\n".join(relevant_chunks)
                prompt = f"Based on the following information, answer the question in simple and easy language for the user to understand. If the information is not sufficient, state that you cannot answer based on the provided documents.\n\nInformation:\n{context}\n\nQuestion: {incoming_msg}\n\nAnswer:"
                
                try:
                    response = gen_model.generate_content(prompt)
                    print(f"Raw LLM response: {response}")
                    answer = response.text
                    print(f"LLM generated answer: {answer}")
                    send_whatsapp_message(from_number, answer)
                except Exception as e:
                    print(f"Error generating answer with LLM: {e}")
                    send_whatsapp_message(from_number, "Sorry, I couldn't generate an answer at this time. Please try again.")
            else:
                send_whatsapp_message(from_number, "I couldn't find any relevant information in the uploaded PDFs for your question.")
        else:
            send_whatsapp_message(from_number, "Please upload a PDF first so I can answer questions about it.")
    else:
        send_whatsapp_message(from_number, "Please send a PDF file or ask a question about an uploaded PDF.")


@app.route('/whatsapp', methods=['POST'])
def whatsapp_webhook():
    # Initialize Twilio MessagingResponse for immediate empty response
    resp = MessagingResponse()
    
    # Extract necessary info from incoming request
    from_number = request.values.get('From', '')
    incoming_msg = request.values.get('Body', '').lower()
    num_media = int(request.values.get('NumMedia', 0))
    
    if num_media > 0:
        media_url = request.values.get(f'MediaUrl{0}') # Assuming only one media per message
        media_content_type = request.values.get(f'MediaContentType{0}')
        
        if media_content_type == 'application/pdf':
            # Send initial acknowledgement immediately
            send_whatsapp_message(from_number, "Document received! Starting to process your PDF in the background. You'll get a confirmation soon.")
            
            # Start background thread for processing
            thread = threading.Thread(target=process_whatsapp_message_in_background, args=(from_number, incoming_msg, num_media, media_url, media_content_type))
            thread.start()
        else:
            send_whatsapp_message(from_number, f"I received a {media_content_type} file. I can only process PDFs at the moment.")
    elif incoming_msg == "hi" or incoming_msg == "hello":
        send_whatsapp_message(from_number, "Hello! Welcome to Bima AI. Please send me your insurance policy PDF.")
    elif incoming_msg: # User sent a text message (a query)
        send_whatsapp_message(from_number, "I've received your question and am searching for an answer. This may take a moment.")
        thread = threading.Thread(target=process_whatsapp_message_in_background, args=(from_number, incoming_msg, num_media))
        thread.start()
    else:
        send_whatsapp_message(from_number, "Please send a PDF file or ask a question about an uploaded PDF.")

    # Return empty TwiML immediately to avoid webhook timeouts
    return str(resp)

if __name__ == '__main__':
    app.run(debug=True)
