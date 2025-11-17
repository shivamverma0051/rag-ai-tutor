# 🚀 Deployment Guide - RAG AI Tutor

This guide provides multiple deployment options for your RAG AI Tutor project.

## 🌐 **Option 1: Deploy to Render (Recommended - Free)**

Render provides free hosting perfect for this project.

### Steps:

1. **Prepare Repository**
   ```bash
   # Make sure all changes are committed
   git add .
   git commit -m "🚀 Ready for deployment"
   git push
   ```

2. **Deploy on Render**
   - Go to [render.com](https://render.com)
   - Sign up/Login with GitHub
   - Click "New+" → "Web Service"
   - Connect your GitHub repository: `shivamverma0051/rag-ai-tutor`
   - Configure settings:
     - **Name**: `rag-ai-tutor`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

3. **Add Environment Variable**
   - In Render dashboard → Environment
   - Add: `GEMINI_API_KEY` = `your_actual_api_key`

4. **Deploy**
   - Click "Create Web Service"
   - Wait 5-10 minutes for deployment
   - Your app will be live at: `https://rag-ai-tutor-xxx.onrender.com`

---

## 🚂 **Option 2: Deploy to Railway**

Railway offers simple deployment with generous free tier.

### Steps:

1. **Deploy on Railway**
   - Go to [railway.app](https://railway.app)
   - Login with GitHub
   - Click "Start a New Project"
   - Select "Deploy from GitHub repo"
   - Choose `shivamverma0051/rag-ai-tutor`

2. **Configure**
   - Railway auto-detects Python project
   - Add environment variable:
     - `GEMINI_API_KEY` = `your_actual_api_key`
   - Deploy automatically starts

3. **Access**
   - Get your URL from Railway dashboard
   - App will be live at: `https://rag-ai-tutor-xxx.up.railway.app`

---

## 🐳 **Option 3: Docker Deployment**

Deploy anywhere using Docker containers.

### Local Docker Testing:

1. **Build Image**
   ```bash
   docker build -t rag-ai-tutor .
   ```

2. **Run Container**
   ```bash
   docker run -d \
     -p 8000:8000 \
     -e GEMINI_API_KEY="your_actual_api_key" \
     --name rag-tutor \
     rag-ai-tutor
   ```

3. **Access**
   - Visit: `http://localhost:8000`

### Docker Compose:

```bash
# Create .env file
echo "GEMINI_API_KEY=your_actual_api_key" > .env

# Start services
docker-compose up -d

# Access at http://localhost:8000
```

---

## ☁️ **Option 4: Deploy to Heroku**

Classic platform with reliable hosting.

### Steps:

1. **Install Heroku CLI**
   - Download from [heroku.com](https://devcenter.heroku.com/articles/heroku-cli)

2. **Deploy**
   ```bash
   # Login to Heroku
   heroku login
   
   # Create app
   heroku create rag-ai-tutor-yourname
   
   # Set environment variable
   heroku config:set GEMINI_API_KEY="your_actual_api_key"
   
   # Deploy
   git push heroku main
   
   # Open app
   heroku open
   ```

---

## 🔧 **Option 5: Deploy to DigitalOcean App Platform**

Professional hosting with good free tier.

### Steps:

1. **Create App**
   - Go to [DigitalOcean Apps](https://cloud.digitalocean.com/apps)
   - Click "Create App"
   - Connect GitHub repository
   - Select `shivamverma0051/rag-ai-tutor`

2. **Configure**
   - **Plan**: Basic ($0/month)
   - **Environment Variables**: Add `GEMINI_API_KEY`
   - **Build Command**: `pip install -r requirements.txt`
   - **Run Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

3. **Deploy**
   - Click "Create Resources"
   - Get your live URL

---

## 🌟 **Recommended: Render Deployment**

For this intern assignment, **Render** is the best choice because:

✅ **Free forever** for small projects  
✅ **Automatic deployments** from GitHub  
✅ **HTTPS included** by default  
✅ **Easy environment variables**  
✅ **No credit card required**  
✅ **Professional URLs**  

### Quick Render Setup:

1. Push your code: `git push`
2. Go to [render.com](https://render.com)
3. Connect GitHub → Select repo → Deploy
4. Add `GEMINI_API_KEY` environment variable
5. **Done!** Your app is live ✨

---

## 🔐 **Security Notes**

- ✅ API key handled via environment variables
- ✅ No sensitive data in repository
- ✅ Production-ready configuration
- ✅ CORS properly configured

## 🎯 **Post-Deployment Checklist**

After deployment, test these features:
- [ ] PDF upload works
- [ ] Chat responses are generated
- [ ] Images display correctly
- [ ] RAG retrieval functions
- [ ] Error handling works

## 📊 **Performance Optimization**

Your deployed app includes:
- Efficient FAISS vector search
- Optimized image serving
- Proper caching headers
- Production logging

---

**🎉 Ready to deploy? Choose your platform and follow the steps above!**