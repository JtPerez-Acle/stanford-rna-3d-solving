#!/bin/bash
# Initialize Git repository for RNA 3D Structure Prediction project

# Print header
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║            RNA 3D Structure Prediction Git Setup                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Initialize Git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "▶ Initializing Git repository..."
    git init
    echo "✓ Git repository initialized"
else
    echo "✓ Git repository already initialized"
fi

# Add all files to staging
echo "▶ Adding files to staging..."
git add .

# Show status
echo "▶ Current Git status:"
git status

# Instructions for committing
echo ""
echo "▶ To commit these changes, run:"
echo "  git commit -m \"Initial commit\""
echo ""
echo "▶ To add a remote repository, run:"
echo "  git remote add origin <repository-url>"
echo ""
echo "▶ To push to the remote repository, run:"
echo "  git push -u origin main"
echo ""
echo "✓ Git setup complete!"
