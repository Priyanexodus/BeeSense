#!/bin/bash

# BeeSense Audio Processing - Deployment Script
# This script builds and deploys the SAM application

set -e  # Exit on error

echo "========================================"
echo "BeeSense Audio Processing Deployment"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}✗ AWS CLI not found. Please install it first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ AWS CLI found${NC}"

# Check if SAM CLI is installed
if ! command -v sam &> /dev/null; then
    echo -e "${RED}✗ SAM CLI not found. Please install it first.${NC}"
    echo "  brew install aws-sam-cli  # macOS"
    echo "  pip install aws-sam-cli   # Python"
    exit 1
fi
echo -e "${GREEN}✓ SAM CLI found${NC}"

# Check if S3 bucket exists
BUCKET_NAME="bee-sense"
if aws s3 ls "s3://${BUCKET_NAME}" 2>&1 | grep -q 'NoSuchBucket'; then
    echo -e "${YELLOW}⚠ S3 bucket '${BUCKET_NAME}' not found${NC}"
    read -p "Do you want to create it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        aws s3 mb "s3://${BUCKET_NAME}"
        echo -e "${GREEN}✓ S3 bucket created${NC}"
    else
        echo -e "${RED}✗ S3 bucket required. Exiting.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ S3 bucket '${BUCKET_NAME}' exists${NC}"
fi

echo ""
echo "========================================"
echo "Step 1: Building SAM Application"
echo "========================================"
sam build

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build successful${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

echo ""
echo "========================================"
echo "Step 2: Deploying to AWS"
echo "========================================"

# Check if samconfig.toml exists
if [ -f "samconfig.toml" ]; then
    echo "Using existing SAM configuration..."
    sam deploy
else
    echo "Running guided deployment..."
    sam deploy --guided
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Deployment successful${NC}"
else
    echo -e "${RED}✗ Deployment failed${NC}"
    exit 1
fi

echo ""
echo "========================================"
echo "Step 3: Retrieving Outputs"
echo "========================================"

STACK_NAME="beesense-audio-stack"

# Get stack outputs
echo "Fetching stack outputs..."
API_URL=$(aws cloudformation describe-stacks \
    --stack-name ${STACK_NAME} \
    --query 'Stacks[0].Outputs[?OutputKey==`AudioChunkApiEndpoint`].OutputValue' \
    --output text 2>/dev/null || echo "")

TABLE_NAME=$(aws cloudformation describe-stacks \
    --stack-name ${STACK_NAME} \
    --query 'Stacks[0].Outputs[?OutputKey==`AudioChunksTableName`].OutputValue' \
    --output text 2>/dev/null || echo "")

echo ""
echo "========================================"
echo "Deployment Complete! 🎉"
echo "========================================"
echo ""
echo "API Gateway Endpoint:"
echo "  ${API_URL}"
echo ""
echo "DynamoDB Table:"
echo "  ${TABLE_NAME}"
echo ""
echo "S3 Bucket:"
echo "  s3://${BUCKET_NAME}"
echo ""
echo "========================================"
echo "Next Steps:"
echo "========================================"
echo ""
echo "1. Update test_chunk_upload.py with your API endpoint:"
echo "   API_ENDPOINT = \"${API_URL}\""
echo ""
echo "2. Test the deployment:"
echo "   python3 test_chunk_upload.py"
echo ""
echo "3. Configure your ESP32 with the following:"
echo "   - API Endpoint: ${API_URL}"
echo "   - Or MQTT Topics:"
echo "     * Chunk: beesense/audio/chunk"
echo "     * Complete: beesense/audio/complete"
echo ""
echo "4. Monitor CloudWatch Logs:"
echo "   sam logs -n AudioChunkReceiverFunction --tail"
echo ""
echo "========================================"

# Save outputs to a file
cat > deployment-outputs.txt << EOF
BeeSense Audio Processing - Deployment Outputs
===============================================

Deployed: $(date)

API Gateway Endpoint:
${API_URL}

DynamoDB Table:
${TABLE_NAME}

S3 Bucket:
${BUCKET_NAME}

MQTT Topics (IoT Core):
- Chunk upload: beesense/audio/chunk
- Completion: beesense/audio/complete

Lambda Functions:
- Chunk Receiver: beesense-audio-chunk-receiver
- HTTP Receiver: beesense-audio-chunk-http
- Audio Stitcher: beesense-audio-stitcher
- Session Completion: beesense-session-completion

S3 Paths:
- Raw chunks: s3://${BUCKET_NAME}/raw-audio-chunks/
- Processed audio: s3://${BUCKET_NAME}/processed-audio/

===============================================
EOF

echo "Deployment outputs saved to: deployment-outputs.txt"
echo ""