# GitHub Repository Setup Script for Windows
# Run this after creating your repository on GitHub

Write-Host "🎓 RAG AI Tutor - GitHub Setup" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "⚠️  IMPORTANT: Before running this script:" -ForegroundColor Yellow
Write-Host "1. Go to https://github.com and create a new repository named 'rag-ai-tutor'" -ForegroundColor White
Write-Host "2. Do NOT initialize with README, .gitignore, or license" -ForegroundColor White
Write-Host "3. Copy the repository URL" -ForegroundColor White
Write-Host ""

$repoUrl = Read-Host "Enter your GitHub repository URL (https://github.com/username/rag-ai-tutor.git)"
Write-Host ""

# Validate URL
if ($repoUrl -notmatch "^https://github\.com/.+/.*\.git$") {
    Write-Host "❌ Invalid GitHub URL format. Please check and try again." -ForegroundColor Red
    exit 1
}

Write-Host "🔗 Setting up remote repository..." -ForegroundColor Green
git remote add origin $repoUrl

Write-Host "🌟 Setting default branch to main..." -ForegroundColor Green
git branch -M main

Write-Host "🚀 Pushing to GitHub..." -ForegroundColor Green
git push -u origin main

Write-Host ""
Write-Host "✅ SUCCESS! Your RAG AI Tutor is now on GitHub!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "1. Go to your GitHub repository" -ForegroundColor White
Write-Host "2. Add repository topics: ai, rag, fastapi, google-gemini, education, python" -ForegroundColor White
Write-Host "3. Update repository description" -ForegroundColor White
Write-Host "4. Star your repository ⭐" -ForegroundColor White
Write-Host ""
$repoUrlClean = $repoUrl -replace "\.git$", ""
Write-Host "🎉 Repository URL: $repoUrlClean" -ForegroundColor Green
Write-Host ""
Write-Host "Happy coding! 🚀" -ForegroundColor Magenta