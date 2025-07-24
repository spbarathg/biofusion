#!/bin/bash

# SIMPLIFIED DEPLOYMENT SCRIPT - LEAN MATHEMATICAL CORE
# =====================================================

set -e

echo "🚀 Deploying Simplified Trading Bot..."

# Configuration
DOCKER_COMPOSE_FILE="simplified-docker-compose.yml"
ENV_FILE=".env.production"

# Check if environment file exists
if [[ ! -f "$ENV_FILE" ]]; then
    echo "❌ Environment file $ENV_FILE not found!"
    echo "💡 Copy config/simplified.env.template to $ENV_FILE and configure it"
    exit 1
fi

# Build and deploy
echo "📦 Building simplified trading bot..."
docker-compose -f $DOCKER_COMPOSE_FILE build

echo "🔄 Starting simplified trading bot..."
docker-compose -f $DOCKER_COMPOSE_FILE up -d

echo "📊 Checking deployment status..."
sleep 10
docker-compose -f $DOCKER_COMPOSE_FILE ps

echo "✅ Simplified trading bot deployed successfully!"
echo "📈 View logs: docker-compose -f $DOCKER_COMPOSE_FILE logs -f simplified-bot"
echo "🔍 Check health: curl http://localhost:8080/health" 