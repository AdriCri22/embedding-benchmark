#!/bin/bash
# TechCorp AI Platform Deployment Script
# Usage: ./deploy.sh [environment] [version]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="${1:-staging}"
VERSION="${2:-latest}"
NAMESPACE="techcorp-${ENVIRONMENT}"
IMAGE="ghcr.io/techcorp/ai-platform"
KUBECONFIG_FILE="${KUBECONFIG:-$HOME/.kube/config}"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Check your kubeconfig."
        exit 1
    fi
    
    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace $NAMESPACE does not exist. Creating..."
        kubectl create namespace "$NAMESPACE"
    fi
    
    log_success "Prerequisites check passed"
}

deploy() {
    local version="$1"
    local image="ghcr.io/techcorp/ai-platform:${version}"
    
    log_info "Deploying version ${version} to ${ENVIRONMENT}..."
    
    # Update deployment image
    kubectl set image deployment/techcorp-ai-platform \
        api="ghcr.io/techcorp/ai-platform:${version}" \
        -n "${NAMESPACE}" \
        --record
    
    # Wait for rollout
    log_info "Waiting for rollout to complete..."
    if kubectl rollout status deployment/techcorp-ai-platform \
        -n "${NAMESPACE}" \
        --timeout=300s; then
        log_success "Deployment completed successfully"
    else
        log_error "Deployment failed or timed out"
        exit 1
    fi
    
    # Verify deployment
    verify_deployment
}

verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check pods are ready
    local ready_pods
    ready_pods=$(kubectl get pods -n "${NAMESPACE}" \
        -l app=techcorp-ai-platform \
        -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' | \
        grep -o "True" | wc -l)
    
    local total_pods
    total_pods=$(kubectl get pods -n "${NAMESPACE}" \
        -l app=techcorp-ai-platform \
        --no-headers | wc -l)
    
    if [[ "$ready_pods" -eq "$total_pods" && "$total_pods" -gt 0 ]]; then
        log_success "All ${total_pods} pods are ready"
    else
        log_error "Only ${ready_pods}/${total_pods} pods are ready"
        return 1
    fi
    
    # Health check
    local service_ip
    service_ip=$(kubectl get service techcorp-ai-platform -n "${NAMESPACE}" -o jsonpath='{.spec.clusterIP}')
    
    if [[ -n "$service_ip" ]]; then
        log_info "Service IP: ${service_ip}"
        # Test health endpoint
        if kubectl run -i --rm --restart=Never curl-test \
            --image=curlimages/curl:latest \
            -- curl -sf "http://techcorp-ai-platform.${NAMESPACE}.svc.cluster.local/health/live" &> /dev/null; then
            log_success "Health check passed"
        else
            log_warning "Health check failed"
        fi
    fi
}

rollback() {
    log_warning "Rolling back deployment..."
    kubectl rollout undo deployment/techcorp-ai-platform -n "${NAMESPACE}"
    kubectl rollout status deployment/techcorp-ai-platform -n "${NAMESPACE}" --timeout=120s
    log_success "Rollback completed"
}

# Main script
main() {
    log_info "Starting deployment to ${ENVIRONMENT} (version: ${VERSION})"
    
    check_prerequisites
    
    case "${2:-deploy}" in
        deploy)
            deploy "${VERSION}"
            ;;
        rollback)
            rollback
            ;;
        verify)
            verify_deployment
            ;;
        *)
            log_error "Unknown command: ${2:-deploy}. Use: deploy, rollback, or verify"
            exit 1
            ;;
    esac
}

# Run main
main "$@"