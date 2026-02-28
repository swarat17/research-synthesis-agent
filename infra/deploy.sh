#!/bin/bash
set -e

REGION=${AWS_REGION:-us-east-1}
STACK_NAME="research-synthesis-agent"
ECR_REPO="research-agent"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}"

echo "==> Building Docker image..."
docker build -t ${ECR_REPO}:latest -f docker/Dockerfile .

echo "==> Logging into ECR..."
aws ecr get-login-password --region ${REGION} | \
  docker login --username AWS --password-stdin "${ECR_URI}"

echo "==> Creating ECR repo if missing..."
aws ecr describe-repositories --repository-names ${ECR_REPO} --region ${REGION} 2>/dev/null || \
  aws ecr create-repository --repository-name ${ECR_REPO} --region ${REGION}

echo "==> Pushing image to ECR..."
docker tag ${ECR_REPO}:latest ${ECR_URI}:latest
docker push ${ECR_URI}:latest

echo "==> Deploying with SAM..."
sam deploy \
  --template-file infra/template.yaml \
  --stack-name ${STACK_NAME} \
  --region ${REGION} \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    OpenAIApiKey="${OPENAI_API_KEY}" \
    AnthropicApiKey="${ANTHROPIC_API_KEY}" \
    PineconeApiKey="${PINECONE_API_KEY}" \
    PineconeIndex="${PINECONE_INDEX}" \
    SupabaseUrl="${SUPABASE_URL}" \
    SupabaseKey="${SUPABASE_KEY}" \
    SemanticScholarApiKey="${SEMANTIC_SCHOLAR_API_KEY:-}" \
    MaxCostPerQuery="${MAX_COST_PER_QUERY:-0.50}"

echo "==> Deployment complete!"
aws cloudformation describe-stacks \
  --stack-name ${STACK_NAME} \
  --region ${REGION} \
  --query "Stacks[0].Outputs" \
  --output table
