# 🎓 RAG-Based AI Tutor with Images

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent AI tutoring system that combines **Retrieval-Augmented Generation (RAG)** with **Google Gemini AI** to provide educational responses enriched with relevant images. Upload PDF documents and get contextual, visual learning experiences with ChatGPT-style streaming responses.

## ✨ Features

### 🧠 **Smart AI Capabilities**
- **RAG Pipeline**: TF-IDF + FAISS vector search for accurate content retrieval
- **Google Gemini Integration**: High-quality educational responses using Gemini 2.0 Flash
- **Contextual Understanding**: Intelligent greeting recognition and appropriate responses
- **Educational Focus**: Specialized prompting for learning-oriented interactions

### 🎨 **Rich User Experience**
- **Drag & Drop PDF Upload**: Seamless document processing
- **Streaming Responses**: ChatGPT-style typing animations
- **Visual Learning**: 8 educational diagrams with smart image matching
- **Responsive Design**: Modern, clean interface that works on all devices
- **Real-time Chat**: Instant responses with loading indicators

## 🏗️ Architecture

### RAG Pipeline Explanation

1. **Text Extraction**: Uses PyPDF2 to extract text from uploaded PDF files
2. **Text Chunking**: Splits documents into overlapping chunks (500 chars with 50 char overlap)
3. **Embedding Generation**: Creates TF-IDF embeddings for all text chunks
4. **Vector Storage**: Uses FAISS for efficient similarity search
5. **Retrieval**: Finds top-k relevant chunks using cosine similarity
6. **Answer Generation**: Constructs responses using retrieved context

### Image Retrieval Logic

The system uses a multi-factor scoring approach to select relevant images:

- **Keyword Matching**: Scores based on query/answer keywords matching image metadata
- **Semantic Similarity**: Analyzes titles and descriptions for content relevance
- **Fallback Strategy**: Ensures an image is always provided for visual learning

Image metadata includes:
- Unique identifiers
- Descriptive keywords
- Educational context
- Visual descriptions

## 📁 Project Structure

```
rag-tutor/
├── main.py                 # FastAPI backend server
├── index.html             # Frontend interface
├── requirements.txt       # Python dependencies
├── sound_images.json     # Image metadata
├── static/               # Static image files
└── README.md             # This file
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone and Navigate

```bash
cd rag-tutor
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Create Static Directory

```bash
mkdir static
```

### Step 4: Add Images (Optional)

Place image files referenced in `sound_images.json` in the `static/` directory:
- bell.png
- sound_waves.png
- frequency_amplitude.png
- human_ear.png
- echo_reflection.png
- musical_instruments.png
- ultrasound_infrasound.png
- doppler_effect.png

### Step 5: Run the Application

```bash
python main.py
```

The application will be available at: http://localhost:8000

## 🎯 Usage

1. **Upload PDF**: Drag and drop or select a PDF file
2. **Wait for Processing**: The system will extract and chunk the text
3. **Start Chatting**: Ask questions about the document content
4. **View Results**: Get answers with relevant images displayed inline

## 📊 API Endpoints

### POST /upload
- **Description**: Upload and process PDF documents
- **Input**: Multipart form data with PDF file
- **Output**: Topic ID and processing status

### POST /chat
- **Description**: Send chat messages and receive AI responses
- **Input**: JSON with message and topic_id
- **Output**: Answer text, relevant image, and source information

### GET /images/{topic_id}
- **Description**: Retrieve image metadata for a topic
- **Output**: List of available images with metadata

### GET /health
- **Description**: System health check
- **Output**: Status and system statistics

## 🧠 Prompts and Answer Generation

The system uses a template-based approach for answer generation:

1. **Context Preparation**: Combines retrieved text chunks
2. **Relevance Scoring**: Ranks chunks by query similarity
3. **Answer Construction**: Uses the most relevant chunk as primary content
4. **Response Formatting**: Structures the answer for clarity

Example prompt structure:
```
Based on the uploaded content:

[Most relevant text chunk with 400 character limit]
```

## 🔧 Configuration

### Text Processing
- **Chunk Size**: 500 characters
- **Chunk Overlap**: 50 characters
- **Retrieval Count**: 3-5 most relevant chunks
- **TF-IDF Features**: 1000 maximum features

### Image Selection
- **Keyword Weight**: 2 points per matching keyword
- **Title/Description Weight**: 1 point per matching word
- **Minimum Threshold**: Always returns best match or fallback

## 🚀 Deployment

### Local Development
```bash
python main.py
```

### Production Deployment
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker (Optional)
Create a Dockerfile:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🧪 Testing

Test the system with sample queries:
- "What is sound?"
- "How do musical instruments produce sound?"
- "Explain the Doppler effect"
- "What is the structure of the human ear?"

## 🔍 Troubleshooting

### Common Issues

1. **PDF Processing Fails**
   - Ensure PDF is text-based (not scanned images)
   - Check file size and format

2. **No Images Displayed**
   - Verify static/ directory exists
   - Check image filenames match metadata
   - Ensure proper file permissions

3. **Empty Responses**
   - Confirm PDF has sufficient text content
   - Check if embeddings were created successfully

### Debug Mode

Enable detailed logging by modifying the logging level:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Optimization

- **Embedding Caching**: Embeddings are stored in memory for fast retrieval
- **FAISS Integration**: Efficient similarity search for large document collections
- **Chunk Overlap**: Ensures important information isn't lost at boundaries
- **Response Truncation**: Limits answer length for better readability

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- FastAPI for the robust web framework
- FAISS for efficient vector similarity search
- TF-IDF for text embedding generation
- PyPDF2 for PDF text extraction

---

**Note**: This is an educational project demonstrating RAG implementation. For production use, consider implementing proper authentication, error handling, and scalability features.