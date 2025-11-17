#!/bin/bash
# GitHub Repository Setup Script
# Run this after creating your repository on GitHub

echo "🎓 RAG AI Tutor - GitHub Setup"
echo "================================="
echo ""
echo "⚠️  IMPORTANT: Before running this script:"
echo "1. Go to https://github.com and create a new repository named 'rag-ai-tutor'"
echo "2. Do NOT initialize with README, .gitignore, or license"
echo "3. Copy the repository URL"
echo ""
read -p "Enter your GitHub repository URL (https://github.com/username/rag-ai-tutor.git): " REPO_URL
echo ""

# Validate URL
if [[ ! $REPO_URL =~ ^https://github\.com/.+/.*\.git$ ]]; then
    echo "❌ Invalid GitHub URL format. Please check and try again."
    exit 1
fi

echo "🔗 Setting up remote repository..."
git remote add origin "$REPO_URL"

echo "🌟 Setting default branch to main..."
git branch -M main

echo "🚀 Pushing to GitHub..."
git push -u origin main

echo ""
echo "✅ SUCCESS! Your RAG AI Tutor is now on GitHub!"
echo ""
echo "📋 Next steps:"
echo "1. Go to your GitHub repository"
echo "2. Add repository topics: ai, rag, fastapi, google-gemini, education, python"
echo "3. Update repository description"
echo "4. Star your repository ⭐"
echo ""
echo "🎉 Repository URL: ${REPO_URL%.git}"
echo ""
echo "Happy coding! 🚀"