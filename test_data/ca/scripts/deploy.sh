#!/bin/bash
# Script de Desplegament de TechCorp AI Platform
# Ús: ./deploy.sh [entorn] [versió]

set -euo pipefail

# Colors per sortida
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # Sense Color

# Valors per defecte
ENTORN="${1:-staging}"
VERSIO="${2:-latest}"
NAMESPACE="techcorp-${ENTORN}"
IMAGE="ghcr.io/techcorp/ai-platform"
KUBECONFIG_FILE="${KUBECONFIG:-$HOME/.kube/config}"

# Funcions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[ÈXIT]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[AVÍS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Verificant prererequisits..."
    
    # Verificar kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl no trobat. Si us plau instal·la kubectl."
        exit 1
    fi
    
    # Verificar connexió kubectl
    if ! kubectl cluster-info &> /dev/null; then
        log_error "No es pot connectar al clúster Kubernetes. Verifica el teu kubeconfig."
        exit 1
    fi
    
    # Verificar que el namespace existeix
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "El namespace $NAMESPACE no existeix. Creant..."
        kubectl create namespace "$NAMESPACE"
    fi
    
    log_success "Verificació de prererequisits completada"
}

deploy() {
    local version="$1"
    local image="ghcr.io/techcorp/ai-platform:${version}"
    
    log_info "Desplegant versió ${version} a ${ENTORN}..."
    
    # Actualitzar imatge del deployment
    kubectl set image deployment/techcorp-ai-platform \
        api="ghcr.io/techcorp/ai-platform:${version}" \
        -n "${NAMESPACE}" \
        --record
    
    # Esperar rollout
    log_info "Esperant que completi el rollout..."
    if kubectl rollout status deployment/techcorp-ai-platform \
        -n "${NAMESPACE}" \
        --timeout=300s; then
        log_success "Desplegament completat amb èxit"
    else
        log_error "El desplegament ha fallat o ha exhaurit el temps"
        exit 1
    fi
    
    # Verificar desplegament
    verify_deployment
}

verify_deployment() {
    log_info "Verificant desplegament..."
    
    # Verificar que els pods estiguin llestos
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
        log_success "Tots els ${total_pods} pods estan llestos"
    else
        log_error "Només ${ready_pods}/${total_pods} pods estan llestos"
        return 1
    fi
    
    # Health check
    local service_ip
    service_ip=$(kubectl get service techcorp-ai-platform -n "${NAMESPACE}" -o jsonpath='{.spec.clusterIP}')
    
    if [[ -n "$service_ip" ]]; then
        log_info "IP del servei: ${service_ip}"
        # Provar endpoint de salut
        if kubectl run -i --rm --restart=Never curl-test \
            --image=curlimages/curl:latest \
            -- curl -sf "http://techcorp-ai-platform.${NAMESPACE}.svc.cluster.local/health/live" &> /dev/null; then
            log_success "Health check passat"
        else
            log_warning "Health check ha fallat"
        fi
    fi
}

rollback() {
    log_warning "Fent rollback del desplegament..."
    kubectl rollout undo deployment/techcorp-ai-platform -n "${NAMESPACE}"
    kubectl rollout status deployment/techcorp-ai-platform -n "${NAMESPACE}" --timeout=120s
    log_success "Rollback completat"
}

# Script principal
main() {
    log_info "Iniciant desplegament a ${ENTORN} (versió: ${VERSIO})"
    
    check_prerequisites
    
    case "${2:-deploy}" in
        deploy)
            deploy "${VERSIO}"
            ;;
        rollback)
            rollback
            ;;
        verify)
            verify_deployment
            ;;
        *)
            log_error "Comanda desconeguda: ${2:-deploy}. Usa: deploy, rollback, o verify"
            exit 1
            ;;
    esac
}

# Executar main
main "$@"