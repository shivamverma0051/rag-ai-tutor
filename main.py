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
# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")  # Use environment variable first
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"

# Validate API key
if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE" or not GEMINI_API_KEY:
    logger.warning("⚠️  GEMINI_API_KEY not configured. Please set environment variable or update main.py")

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

def retrieve_relevant_chunks(query: str, topic_id: str, k: int = 5) -> List[str]:
    """Enhanced retrieval that prioritizes direct question matching"""
    if topic_id not in text_chunks:
        return []
    
    chunks = text_chunks[topic_id]
    vectorizer = vectorizers[topic_id]
    
    # Dramatically improve query preprocessing for better matching
    query_lower = query.lower().strip()
    
    # For basic definition questions, prioritize fundamental concepts
    if any(word in query_lower for word in ['what is', 'define', 'definition']):
        if 'what is sound' in query_lower or 'define sound' in query_lower:
            # Focus on basic sound definition, not specific applications
            enhanced_query = "sound definition vibration energy waves medium air matter longitudinal pressure"
        elif 'what is frequency' in query_lower:
            enhanced_query = "frequency definition vibration cycles per second hertz pitch"
        elif 'what is amplitude' in query_lower:
            enhanced_query = "amplitude definition loudness volume intensity sound wave height"
        elif 'what is' in query_lower:
            concept = query_lower.replace('what is', '').replace('?', '').strip()
            enhanced_query = f"{concept} definition basic fundamental concept explanation"
        else:
            enhanced_query = f"{query} definition explanation"
    else:
        enhanced_query = query
    
    # Use enhanced query for embedding
    query_embedding = vectorizer.transform([enhanced_query])
    
    # Use FAISS for similarity search
    index = faiss_indexes[topic_id]
    similarities, indices = index.search(query_embedding.toarray().astype(np.float32), k)
    
    # Get chunks with similarity scores
    chunk_scores = list(zip(indices[0], similarities[0]))
    
    # For definition questions, also do keyword-based filtering
    if any(word in query_lower for word in ['what is', 'define']):
        main_concept = None
        if 'what is sound' in query_lower:
            main_concept = 'sound'
        elif 'what is' in query_lower:
            main_concept = query_lower.split('what is')[-1].strip().replace('?', '')
        
        if main_concept:
            # Score chunks based on how well they define the concept
            definition_scores = []
            for idx, sim_score in chunk_scores:
                if idx < len(chunks):
                    chunk = chunks[idx].lower()
                    def_score = 0
                    
                    # Boost chunks that contain definition patterns
                    definition_patterns = [
                        f"{main_concept} is",
                        f"{main_concept} are",
                        f"definition of {main_concept}",
                        f"{main_concept} refers to",
                        f"{main_concept} can be defined",
                        f"what is {main_concept}",
                        f"{main_concept}:",
                        f"sound waves",
                        f"longitudinal waves",
                        f"compression and rarefaction"
                    ]
                    
                    for pattern in definition_patterns:
                        if pattern in chunk:
                            def_score += 2.0
                    
                    # Penalize chunks about specific applications without definitions
                    penalty_patterns = [
                        "reflection", "echo", "doppler", "interference", 
                        "experiment", "measurement", "calculation"
                    ]
                    
                    for pattern in penalty_patterns:
                        if pattern in chunk and main_concept not in chunk[:100]:  # Not in first 100 chars
                            def_score -= 1.0
                    
                    # Boost chunks that mention the concept early
                    if main_concept in chunk[:200]:
                        def_score += 1.0
                    
                    # Combine similarity score with definition score
                    combined_score = sim_score + def_score
                    definition_scores.append((idx, combined_score))
            
            # Sort by combined score and take top chunks
            definition_scores.sort(key=lambda x: x[1], reverse=True)
            relevant_chunks = []
            for idx, score in definition_scores[:k]:
                if idx < len(chunks):
                    relevant_chunks.append(chunks[idx])
            
            return relevant_chunks
    
    # Default behavior for non-definition questions
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
        # Check query complexity
        query_words = query.split()
        is_simple_query = len(query_words) < 5
        
        # Prepare context from retrieved chunks
        context = "\\n\\n".join(relevant_chunks[:3])  # Use top 3 chunks
        
        # Create appropriate prompt based on query complexity
        if is_simple_query:
            prompt = f"""You are an educational AI tutor. The student has asked a specific question. You MUST answer their EXACT question directly and completely.

**CRITICAL INSTRUCTION**: The student asked: "{query}"
You MUST provide a COMPLETE, STANDALONE explanation that fully answers this question. The student will see an educational image AFTER your text explanation, so your answer must be comprehensive on its own.

**Context from PDF:**
{context}

**Student's EXACT Question:** {query}

Provide a COMPLETE, DETAILED answer in this EXACT format:

📚 **Answer**
• [Clear, complete definition or explanation that fully answers the question]
• [Essential characteristics, properties, or key concepts]
• [Important details and examples that enhance understanding]
• [Any additional relevant information that completes the explanation]

**STRICT RULES:**
1. Provide a COMPLETE answer - don't assume they'll see the image first
2. Answer ONLY the question that was asked
3. If they ask "what is sound" - define what sound IS comprehensively
4. If they ask "what is [concept]" - explain that concept thoroughly
5. Use ONLY the provided context that relates to their specific question
6. Make your text explanation self-sufficient and complete
7. Include key facts, definitions, and examples in your TEXT answer"""
        else:
            prompt = f"""You are an expert educational AI tutor. The student has asked a specific question and you MUST provide a COMPLETE, COMPREHENSIVE answer.

**CRITICAL INSTRUCTION**: The student asked: "{query}"
You MUST provide a thorough, complete explanation. A relevant educational image will be shown AFTER your text, so your explanation must be comprehensive and self-sufficient.

**Context from PDF:**
{context}

**Student's EXACT Question:** {query}

Provide a comprehensive, complete answer in this EXACT format:

📚 **Complete Answer**
• [Thorough primary explanation that fully addresses what they asked]
• [Key scientific facts, principles, or definitions]
• [Important details and characteristics]
• [Examples or applications that enhance understanding]

💡 **Key Points**
• [Specific information from the context about their topic]
• [Scientific facts, data, or measurements if available]
• [Real-world examples or practical applications]
• [Additional important details that complete the answer]

**ABSOLUTE REQUIREMENTS:**
1. Provide a COMPLETE, COMPREHENSIVE text explanation
2. Your answer MUST directly address what they asked in detail
3. Don't say "as shown in the image" - make your text self-sufficient
4. If they ask "what is X" - explain what X IS thoroughly with all key details
5. Use ONLY context information that helps answer their specific question
6. Include definitions, characteristics, examples, and applications in your TEXT
7. Make your explanation complete enough that someone could understand without seeing any images"""

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
                "temperature": 0.3,  # Lower temperature for more focused responses
                "maxOutputTokens": 800,  # Reasonable limit
                "topP": 0.9,
                "topK": 20
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

def handle_simple_inputs(message: str) -> str:
    """Handle greetings and simple conversational inputs"""
    message_lower = message.lower().strip()
    
    # Greetings
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
    if any(greeting in message_lower for greeting in greetings):
        return "👋 **Hello! Welcome to your AI Tutor!**\n\n• Upload a PDF document to get started\n• Ask questions about the content\n• I'll provide answers with relevant images\n\nHow can I help you learn today?"
    
    # Thanks
    thanks = ['thanks', 'thank you', 'thx']
    if any(thank in message_lower for thank in thanks):
        return "🌟 **You're welcome!**\n\nHappy to help you learn! Feel free to ask more questions."
    
    # Help requests
    help_words = ['help', 'what can you do', 'how to use']
    if any(help_word in message_lower for help_word in help_words):
        return "📚 **How to use your AI Tutor:**\n\n• **Upload PDF**: Drag & drop or click to upload educational documents\n• **Ask Questions**: Type questions about the uploaded content\n• **Get Answers**: Receive detailed explanations with bullet points\n• **View Images**: See relevant educational diagrams\n\nReady to start learning?"
    
    # Short/unclear inputs
    if len(message_lower) < 3 or message_lower in ['ok', 'yes', 'no', 'hmm', 'what']:
        return "🤔 **I'm here to help!**\n\nPlease ask a specific question about your uploaded document, or upload a PDF to get started."
    
    return None  # Not a simple input, proceed with normal processing

def generate_fallback_answer(query: str, relevant_chunks: List[str]) -> str:
    """Generate fallback answer when Gemini API fails - with better question matching"""
    if not relevant_chunks:
        return "❓ **Information Not Available**\\n\\nI don't have enough information to answer that question. Please make sure you've uploaded a relevant PDF document."
    
    # Find most relevant chunk based on the specific question
    query_lower = query.lower()
    query_words = [word.strip('.,?!') for word in query_lower.split()]
    
    best_chunk = ""
    best_score = 0
    
    # Special handling for definition questions
    if 'what is' in query_lower or 'define' in query_lower:
        concept = None
        if 'what is sound' in query_lower:
            concept = 'sound'
        elif 'what is' in query_lower:
            concept = query_lower.split('what is')[-1].strip().replace('?', '')
        
        if concept:
            for chunk in relevant_chunks:
                chunk_lower = chunk.lower()
                score = 0
                
                # High score for chunks that define the concept
                if f"{concept} is" in chunk_lower or f"{concept} are" in chunk_lower:
                    score += 10
                
                # Boost for chunks that mention the concept early
                if concept in chunk_lower[:100]:
                    score += 5
                
                # Boost for definition-like content
                definition_indicators = ['definition', 'refers to', 'can be defined as', 'is a type of']
                for indicator in definition_indicators:
                    if indicator in chunk_lower:
                        score += 3
                
                # Penalize chunks about specific applications without basic definition
                if any(word in chunk_lower for word in ['experiment', 'reflection', 'calculation']) and concept not in chunk_lower[:150]:
                    score -= 5
                
                if score > best_score:
                    best_score = score
                    best_chunk = chunk
    
    # Fallback to basic keyword matching
    if not best_chunk:
        for chunk in relevant_chunks:
            chunk_lower = chunk.lower()
            score = sum(1 for word in query_words if word in chunk_lower)
            
            if score > best_score:
                best_score = score
                best_chunk = chunk
    
    if best_chunk:
        # Create a focused answer based on the question
        if 'what is' in query_lower:
            concept = query_lower.replace('what is', '').strip().replace('?', '')
            answer_parts = [
                f"📚 **Answer about {concept.title()}**",
                "",
                f"Based on the uploaded document:",
                f"• {best_chunk[:200]}..." if len(best_chunk) > 200 else f"• {best_chunk}",
                "",
                "💡 **Note**: This is extracted directly from your uploaded PDF content."
            ]
        else:
            answer_parts = [
                "📚 **Answer from Document**",
                "",
                f"• {best_chunk[:300]}..." if len(best_chunk) > 300 else f"• {best_chunk}",
                "",
                "💡 **Source**: Information extracted from your uploaded PDF."
            ]
        
        return "\\n".join(answer_parts)
    
    return "❓ **Limited Information**\\n\\nThe uploaded document contains related information but may not directly answer your specific question. Please try rephrasing your question or uploading a more relevant document."
    
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
        with open('index.html', 'r', encoding='utf-8') as f:
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
        # Handle simple greetings and casual inputs
        greeting_responses = handle_simple_inputs(message.message)
        if greeting_responses:
            return ChatResponse(
                answer=greeting_responses,
                image=None,
                sources=[]
            )
        
        if not message.topic_id or message.topic_id not in text_chunks:
            return ChatResponse(
                answer="Please upload a PDF document first to start the conversation.",
                image=None,
                sources=[]
            )
        
        # Retrieve relevant chunks for educational queries
        relevant_chunks = retrieve_relevant_chunks(message.message, message.topic_id)
        
        # Generate answer using Gemini
        answer = generate_answer(message.message, relevant_chunks)
        
        # Find relevant image only for educational content
        image = find_relevant_image(message.message, answer) if relevant_chunks else None
        
        return ChatResponse(
            answer=answer,
            image=image,
            sources=[f"Chunk {i+1}" for i in range(len(relevant_chunks))] if relevant_chunks else []
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
    
    # Get port from environment variable (for deployment) or default to 8000
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"🚀 Starting RAG AI Tutor on {host}:{port}")
    logger.info(f"🔑 API Key configured: {'✅ Yes' if GEMINI_API_KEY and GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY_HERE' else '❌ No'}")
    
    uvicorn.run(app, host=host, port=port)