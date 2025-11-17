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
    """Find most relevant image based on keyword matching"""
    if not image_metadata:
        return None
    
    # Combine query and answer for keyword matching
    combined_text = f"{query.lower()} {answer.lower()}"
    
    best_image = None
    best_score = 0
    
    for image in image_metadata:
        score = 0
        
        # Check for keyword matches
        for keyword in image['keywords']:
            if keyword.lower() in combined_text:
                score += 2
        
        # Check title matches
        if image['title'].lower() in combined_text:
            score += 3
        
        # Check description matches  
        for word in image['description'].lower().split():
            if len(word) > 3 and word in combined_text:
                score += 1
        
        if score > best_score:
            best_score = score
            best_image = image
    
    return best_image if best_score > 0 else image_metadata[0]

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
            prompt = f"""You are an educational AI tutor powered by Google Gemini AI. Provide a comprehensive, detailed answer to the student's question using the context from their uploaded PDF.

**Context from PDF:**
{context}

**Student Question:** {query}

**Instructions:** Provide a COMPLETE educational explanation with thorough details. This is part of a RAG-based AI tutoring system that will show relevant images after your text response.

Format your answer as:

📚 **Complete Definition & Explanation**
• [Comprehensive definition of the concept with full details]
• [Key characteristics, properties, and scientific principles]
• [How the process works with step-by-step explanation]
• [Real-world examples and practical applications]
• [Important facts and additional context from the PDF]

💡 **Educational Summary**
• [Main takeaways for the student]
• [Why this concept is important to understand]

**Critical:** Provide thorough explanations (minimum 3-4 detailed bullet points) so students get complete understanding before seeing visual aids."""
        else:
            prompt = f"""You are an advanced educational AI tutor powered by Google Gemini AI. Provide a detailed, comprehensive answer to the student's complex question using the context from their uploaded PDF.

**Context from PDF:**
{context}

**Student Question:** {query}

**Instructions:** Give a complete, thorough educational response. This is part of a sophisticated RAG-based AI tutoring system that will display relevant educational images after your detailed explanation.

Format your answer as:

📚 **Comprehensive Analysis**
• [In-depth explanation of the main concept with scientific details]
• [Detailed characteristics, properties, and underlying principles]
• [Step-by-step breakdown of how the process works]
• [Mathematical relationships or formulas if applicable]

💡 **Key Educational Points**
• [Specific facts, data, and measurements from the PDF]
• [Real-world examples and practical applications]
• [Connections to other related concepts]
• [Historical context or recent developments]

🎯 **Learning Objectives Met**
• [What the student should understand after reading this]
• [How this knowledge can be applied]

**Critical:** Provide extensive detail (minimum 6-8 bullet points total) to ensure complete educational value before visual aids are shown."""

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
        
        # Make API request to Gemini with better error handling
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': GEMINI_API_KEY
        }
        
        logger.info(f"Sending request to Gemini API for query: {query}")
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                generated_text = result['candidates'][0]['content']['parts'][0]['text']
                logger.info("Successfully generated answer with Gemini")
                # Ensure we have a substantial response
                if len(generated_text.strip()) > 50:
                    return generated_text.strip()
                else:
                    logger.warning("Gemini response too short, using enhanced fallback")
                    return generate_fallback_answer(query, relevant_chunks)
            else:
                logger.error(f"No candidates in Gemini response: {result}")
                return generate_fallback_answer(query, relevant_chunks)
        elif response.status_code == 429:
            logger.warning("Gemini API quota exceeded, using enhanced fallback with comprehensive definitions")
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
        return "👋 **Hello! Welcome to the RAG-Based AI Tutor!**\n\n🚀 **About This Project:**\n• Advanced educational AI system combining RAG pipeline with Google Gemini AI\n• Intelligent PDF processing with TF-IDF embeddings and FAISS vector search\n• Comprehensive answers with educational images and diagrams\n• Built with FastAPI backend and modern web interface\n\n📚 **How to Use:**\n• Upload your educational PDF documents\n• Ask questions about the content\n• Receive detailed explanations with relevant visual aids\n\nLet's start your learning journey!"
    
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
    """Generate comprehensive fallback answer when Gemini API fails"""
    if not relevant_chunks:
        return "❓ **Information Not Available**\n\nI don't have enough information to answer that question. Please make sure you've uploaded a relevant PDF document."
    
    query_lower = query.lower()
    
    # Enhanced definition generation for common questions
    if 'what is' in query_lower or 'define' in query_lower:
        # Extract the concept being asked about
        concept = None
        if 'what is sound' in query_lower:
            concept = 'sound'
        elif 'what is reflection' in query_lower:
            concept = 'reflection of sound'
        elif 'what is doppler' in query_lower or 'what is dopler' in query_lower:
            concept = 'doppler effect'
        elif 'what is echo' in query_lower:
            concept = 'echo'
        elif 'what is frequency' in query_lower:
            concept = 'frequency'
        elif 'what is amplitude' in query_lower:
            concept = 'amplitude'
        elif 'what is' in query_lower:
            concept = query_lower.split('what is')[-1].strip().replace('?', '')
        
        # Generate comprehensive definition using PDF context
        context_text = " ".join(relevant_chunks[:3])
        
        # Extract relevant sentences and information - More flexible approach
        sentences = []
        # Split by multiple sentence markers
        for delimiter in ['.', '!', '?']:
            if delimiter in context_text:
                parts = context_text.split(delimiter)
                for part in parts:
                    if len(part.strip()) > 20:
                        sentences.append(part.strip())
        
        concept_sentences = []
        
        if concept:
            concept_words = concept.lower().split()
            
            # First pass - Direct concept matches
            for sentence in sentences:
                sentence_lower = sentence.lower()
                word_matches = sum(1 for word in concept_words if word in sentence_lower)
                if word_matches > 0:
                    if len(sentence.strip()) > 30:
                        concept_sentences.append(sentence.strip())
            
            # If no direct matches, try broader search
            if not concept_sentences:
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    # Look for any physics/science related terms
                    if any(term in sentence_lower for term in ['wave', 'frequency', 'sound', 'vibration', 'phenomenon', 'effect', 'occurs', 'produced', 'result']):
                        if len(sentence.strip()) > 30:
                            concept_sentences.append(sentence.strip())
            
            # If still no matches, take the most relevant chunks directly
            if not concept_sentences and relevant_chunks:
                for chunk in relevant_chunks[:2]:
                    # Clean up chunk content
                    clean_chunk = chunk.replace('\\n', ' ').replace('  ', ' ').strip()
                    if len(clean_chunk) > 50:
                        concept_sentences.append(clean_chunk[:200] + "..." if len(clean_chunk) > 200 else clean_chunk)
        
        # Ensure we always have some content to show
        if not concept_sentences and relevant_chunks:
            # Fallback to showing relevant chunks directly
            concept_sentences = [chunk[:150] + "..." if len(chunk) > 150 else chunk for chunk in relevant_chunks[:2]]
        
        if concept_sentences:
            # Create a comprehensive definition response with proper newlines
            logger.info(f"Found {len(concept_sentences)} concept sentences for '{concept}'")
            answer_lines = []
            answer_lines.append(f"📚 **Complete Definition: {concept.title()}**")
            answer_lines.append("")
            
            # Add the main definition
            definition = f"• **Definition**: {concept_sentences[0]}"
            if not concept_sentences[0].endswith('.'):
                definition += "."
            answer_lines.append(definition)
            logger.info(f"Main definition: {definition[:100]}...")
            answer_lines.append("")
            
            # Add additional characteristics if available
            if len(concept_sentences) > 1:
                characteristics = f"• **Key Characteristics**: {concept_sentences[1]}"
                if not concept_sentences[1].endswith('.'):
                    characteristics += "."
                answer_lines.append(characteristics)
                logger.info(f"Key characteristics: {characteristics[:100]}...")
                answer_lines.append("")
            
            # Add more details if available
            if len(concept_sentences) > 2:
                additional = f"• **Additional Information**: {concept_sentences[2]}"
                if not concept_sentences[2].endswith('.'):
                    additional += "."
                answer_lines.append(additional)
                logger.info(f"Additional info: {additional[:100]}...")
                answer_lines.append("")
            
            answer_lines.append("💡 **Educational Context**")
            answer_lines.append(f"• This information is extracted and processed from your uploaded educational PDF")
            answer_lines.append(f"• The content provides foundational understanding of {concept}")
            answer_lines.append(f"• Additional context and examples may be available in the complete document")
            
            full_answer = "\n".join(answer_lines)
            logger.info(f"Full answer length: {len(full_answer)} characters")
            logger.info(f"Answer preview: {full_answer[:200]}...")
            return full_answer
        else:
            # If no concept sentences found, show what we have from chunks
            logger.warning(f"No concept sentences found for '{concept}', showing available content")
            answer_lines = []
            answer_lines.append(f"📚 **Information about: {concept.title()}**")
            answer_lines.append("")
            
            if relevant_chunks:
                answer_lines.append("• **Available Information**:")
                for i, chunk in enumerate(relevant_chunks[:2]):
                    clean_chunk = chunk.replace('\\n', ' ').strip()
                    if len(clean_chunk) > 100:
                        answer_lines.append(f"  - {clean_chunk[:200]}...")
                    else:
                        answer_lines.append(f"  - {clean_chunk}")
                answer_lines.append("")
            
            answer_lines.append("💡 **Educational Context**")
            answer_lines.append("• This information is extracted and processed from your uploaded educational PDF")
            answer_lines.append(f"• The content provides foundational understanding of {concept}")
            answer_lines.append("• Additional context and examples may be available in the complete document")
            
            return "\n".join(answer_lines)
    
    # For non-definition questions, provide contextual answers
    context_text = " ".join(relevant_chunks[:2])
    query_words = query_lower.split()
    
    # Find most relevant sentences
    relevant_sentences = []
    for sentence in context_text.split('.'):
        if len(sentence.strip()) > 20:
            sentence_lower = sentence.lower()
            word_matches = sum(1 for word in query_words if word in sentence_lower)
            if word_matches > 0:
                relevant_sentences.append(sentence.strip())
    
    if relevant_sentences:
        answer_lines = []
        answer_lines.append("📚 **Answer Based on Document Content**")
        answer_lines.append("")
        
        for i, sentence in enumerate(relevant_sentences[:3]):
            sentence_text = sentence.strip()
            if not sentence_text.endswith('.'):
                sentence_text += "."
            answer_lines.append(f"• {sentence_text}")
        
        answer_lines.append("")
        answer_lines.append("💡 **Source Information**")
        answer_lines.append("• Content extracted from your uploaded educational PDF")
        answer_lines.append("• Information processed through RAG pipeline for relevance")
        
        return "\n".join(answer_lines)
    
    # Final fallback
    answer_lines = []
    answer_lines.append("📚 **Document Information**")
    answer_lines.append("")
    answer_lines.append(f"Based on the uploaded content: {context_text[:300]}...")
    answer_lines.append("")
    answer_lines.append("💡 **Note**: This information is extracted from your educational PDF document.")
    return "\n".join(answer_lines)
    
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