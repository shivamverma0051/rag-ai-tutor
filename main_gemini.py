"""
Fixed main.py with proper Gemini integration
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import PyPDF2
import requests
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import os
import re
from typing import List, Dict, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyB_BH7I-MTpP26oOiEpqCekpe_E_Yi3gwQ"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

app = FastAPI(title="RAG-Based AI Tutor", description="AI Tutor with PDF RAG and Image Retrieval")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Data models
class ChatMessage(BaseModel):
    message: str
    topic_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    image: Optional[Dict] = None
    sources: List[str] = []

# Global storage
text_chunks = {}
embeddings_db = {}
image_metadata = []
vectorizers = {}
faiss_indexes = {}

# Load image metadata
try:
    with open('sound_images.json', 'r') as f:
        image_metadata = json.load(f)
except FileNotFoundError:
    logger.warning("Image metadata file not found")
    image_metadata = []

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
        raise HTTPException(status_code=400, detail="Error processing PDF file")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if len(chunk.strip()) > 20]

def create_embeddings(chunks: List[str]) -> np.ndarray:
    """Create TF-IDF embeddings for text chunks"""
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    embeddings = vectorizer.fit_transform(chunks)
    return embeddings.toarray(), vectorizer

def retrieve_relevant_chunks(query: str, topic_id: str, k: int = 3) -> List[str]:
    """Retrieve most relevant text chunks using cosine similarity"""
    if topic_id not in embeddings_db:
        return []
    
    vectorizer = vectorizers[topic_id]
    query_embedding = vectorizer.transform([query])
    
    # Use FAISS for similarity search
    index = faiss_indexes[topic_id]
    similarities, indices = index.search(query_embedding.toarray().astype(np.float32), k)
    
    relevant_chunks = []
    for idx in indices[0]:
        if idx < len(text_chunks[topic_id]):
            relevant_chunks.append(text_chunks[topic_id][idx])
    
    return relevant_chunks

def find_relevant_image(query: str, answer: str) -> Optional[Dict]:
    """Find most relevant image based on enhanced keyword matching and content analysis"""
    if not image_metadata:
        return None
    
    query_lower = query.lower()
    answer_lower = answer.lower()
    combined_text = f"{query_lower} {answer_lower}"
    
    # Enhanced scoring system
    image_scores = []
    
    for image in image_metadata:
        score = 0
        keywords = [kw.lower() for kw in image['keywords']]
        title_words = [word.lower() for word in image['title'].split()]
        desc_words = [word.lower() for word in image['description'].split()]
        
        # Primary keyword matches (high priority)
        for keyword in keywords:
            if keyword in combined_text:
                # Boost score for exact keyword matches
                score += 5
                # Extra boost for query-specific matches
                if keyword in query_lower:
                    score += 3
        
        # Title word matches (medium priority)
        for word in title_words:
            if len(word) > 3 and word in combined_text:
                score += 2
                if word in query_lower:
                    score += 2
        
        # Description word matches (lower priority)
        for word in desc_words:
            if len(word) > 4 and word in combined_text:
                score += 1
        
        # Contextual scoring for specific topics
        topic_boosters = {
            'bell': ['bell', 'ring', 'school', 'vibrat'],
            'wave': ['wave', 'propagat', 'compress', 'sound'],
            'instrument': ['music', 'guitar', 'drum', 'flute', 'sitar'],
            'vocal': ['voice', 'speak', 'talk', 'vocal', 'throat'],
            'frequency': ['frequency', 'pitch', 'hz', 'hertz', 'range'],
            'echo': ['echo', 'reflect', 'bounce', 'experiment'],
            'doppler': ['doppler', 'moving', 'motion', 'change'],
            'hearing': ['hear', 'ear', 'listen', 'audible', 'range']
        }
        
        for topic, boost_words in topic_boosters.items():
            if any(word in combined_text for word in boost_words):
                if topic in image['filename'] or any(boost_word in image['title'].lower() for boost_word in boost_words):
                    score += 4
        
        image_scores.append((image, score))
    
    # Sort by score and return best match
    image_scores.sort(key=lambda x: x[1], reverse=True)
    
    if image_scores[0][1] > 0:
        return image_scores[0][0]
    
    # Fallback: return most educationally relevant image
    fallback_priorities = ['bell', 'sound_waves', 'musical_instruments', 'frequency_amplitude']
    for priority in fallback_priorities:
        for image in image_metadata:
            if priority in image['filename']:
                return image
    
    # Final fallback
    return image_metadata[0] if image_metadata else None

def generate_answer(query: str, relevant_chunks: List[str]) -> str:
    """Generate well-formatted answer using Gemini AI with retrieved context"""
    if not relevant_chunks:
        return "❓ **Information Not Available**\\n\\nI don't have enough information to answer that question. Please make sure you've uploaded a relevant PDF document that contains information about your query."
    
    try:
        # Prepare context from retrieved chunks
        context = "\\n\\n".join(relevant_chunks[:3])  # Use top 3 chunks
        
        # Create a detailed prompt for Gemini
        prompt = f"""You are an educational AI tutor specializing in physics and sound. Based on the provided context from an uploaded PDF document, answer the student's question in a clear, structured format.

**Context from PDF:**
{context}

**Student Question:** {query}

Please provide a comprehensive answer in this EXACT format:

📚 **Main Answer**
• [Key point 1 explaining the main concept]
• [Key point 2 with specific details]
• [Key point 3 with additional explanation]

💡 **Key Points**
• [Important fact 1 from the context]
• [Important fact 2 with specific data/numbers if available]
• [Important fact 3 with practical application]

🎓 **Educational Context**
• [How this relates to physics principles]
• [Real-world applications or examples]

**Rules:**
1. Use ONLY information from the provided context
2. Format with bullet points as shown above
3. Keep each bullet point concise (1-2 sentences)
4. Include specific details, numbers, or examples when available
5. Make it educational and easy to understand
6. If the context doesn't contain enough information, say so clearly
"""

        # Prepare request payload
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1000,
                "topP": 0.8,
                "topK": 40
            }
        }
        
        # Make API request to Gemini
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': GEMINI_API_KEY
        }
        
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                generated_text = result['candidates'][0]['content']['parts'][0]['text']
                logger.info("Successfully generated answer with Gemini")
                return generated_text.strip()
            else:
                logger.error(f"No candidates in Gemini response: {result}")
                return generate_fallback_answer(query, relevant_chunks)
        else:
            logger.error(f"Gemini API error: {response.status_code} - {response.text}")
            return generate_fallback_answer(query, relevant_chunks)
            
    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        return generate_fallback_answer(query, relevant_chunks)

def generate_fallback_answer(query: str, relevant_chunks: List[str]) -> str:
    """Generate fallback answer when Gemini API fails"""
    if not relevant_chunks:
        return "❓ **Information Not Available**\\n\\nI don't have enough information to answer that question."
    
    # Find most relevant chunk
    query_lower = query.lower()
    query_words = [word.strip('.,?!') for word in query_lower.split()]
    
    best_chunk = ""
    best_score = 0
    
    for chunk in relevant_chunks:
        chunk_lower = chunk.lower()
        score = sum(2 if word in chunk_lower else 0 for word in query_words)
        if score > best_score:
            best_score = score
            best_chunk = chunk
    
    primary_content = best_chunk if best_chunk else relevant_chunks[0]
    
    # Create formatted answer with bullet points
    answer_parts = []
    answer_parts.append("📚 **Main Answer**")
    
    # Split content into sentences for bullet points
    sentences = [s.strip() for s in primary_content.replace('•', '').split('.') if s.strip() and len(s.strip()) > 10]
    
    # Create main explanation points
    main_points = []
    for i, sentence in enumerate(sentences[:3]):
        if sentence:
            clean_sentence = sentence.strip()
            if not clean_sentence.endswith('.'):
                clean_sentence += '.'
            main_points.append(f"• {clean_sentence}")
    
    answer_parts.extend(main_points)
    
    # Add key facts from other chunks
    if len(relevant_chunks) > 1:
        key_facts = []
        for chunk in relevant_chunks[1:3]:
            chunk_sentences = [s.strip() for s in chunk.split('.') if s.strip()]
            for sentence in chunk_sentences:
                if any(word in sentence.lower() for word in query_words) and len(sentence.strip()) > 15:
                    clean_sentence = sentence.strip()
                    if not clean_sentence.endswith('.'):
                        clean_sentence += '.'
                    key_facts.append(f"• {clean_sentence}")
                    break
        
        if key_facts:
            answer_parts.append("\\n💡 **Key Points**")
            answer_parts.extend(key_facts[:3])
    
    # Add educational context
    context_mappings = {
        "sound": "• This relates to acoustics and wave physics principles.\\n• Understanding sound helps explain many natural phenomena.",
        "frequency": "• This involves wave properties and hearing mechanisms.\\n• Frequency determines how we perceive pitch and tone.",
        "vibration": "• This demonstrates oscillatory motion and energy transfer.\\n• Vibration is the fundamental source of all sound production.",
        "wave": "• This illustrates wave propagation and physics principles.\\n• Wave behavior explains how energy travels through matter.",
        "music": "• This explores the science behind musical acoustics.\\n• Musical instruments demonstrate physics in everyday life.",
        "hearing": "• This covers auditory perception and sound processing.\\n• Human hearing involves complex physiological mechanisms."
    }
    
    for keyword, context in context_mappings.items():
        if keyword in query_lower:
            answer_parts.append("\\n🎓 **Educational Context**")
            answer_parts.append(context)
            break
    
    return "\\n".join(answer_parts)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    try:
        with open('index.html', 'r') as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Index.html not found</h1>")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Extract text from PDF and create embeddings"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Extract text
        text = extract_text_from_pdf(file.file)
        
        if len(text.strip()) < 100:
            raise HTTPException(status_code=400, detail="PDF appears to be empty or has insufficient text")
        
        # Create chunks
        chunks = chunk_text(text)
        
        if len(chunks) == 0:
            raise HTTPException(status_code=400, detail="Could not create text chunks from PDF")
        
        # Generate topic ID
        topic_id = f"topic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create embeddings
        embeddings, vectorizer = create_embeddings(chunks)
        
        # Store in FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype(np.float32))
        
        # Store globally
        text_chunks[topic_id] = chunks
        embeddings_db[topic_id] = embeddings
        vectorizers[topic_id] = vectorizer
        faiss_indexes[topic_id] = index
        
        logger.info(f"Successfully processed PDF with {len(chunks)} chunks for topic {topic_id}")
        
        return {
            "topic_id": topic_id,
            "chunks_count": len(chunks),
            "message": "PDF processed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Handle chat messages and return AI responses with images"""
    try:
        if not message.topic_id or message.topic_id not in text_chunks:
            return ChatResponse(
                answer="Please upload a PDF document first to start the conversation.",
                image=None,
                sources=[]
            )
        
        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(message.message, message.topic_id)
        
        # Generate answer using Gemini
        answer = generate_answer(message.message, relevant_chunks)
        
        # Find relevant image
        image = find_relevant_image(message.message, answer)
        
        return ChatResponse(
            answer=answer,
            image=image,
            sources=[f"Chunk {i+1}" for i in range(len(relevant_chunks))]
        )
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/images/{topic_id}")
async def get_images(topic_id: str):
    """Get image metadata for a topic"""
    return {"topic_id": topic_id, "images": image_metadata}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "topics_loaded": len(text_chunks),
        "images_available": len(image_metadata),
        "gemini_api": "configured"
    }

if __name__ == "__main__":
    import uvicorn
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)