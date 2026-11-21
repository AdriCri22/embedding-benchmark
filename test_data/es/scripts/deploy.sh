#!/bin/bash
# Script de Despliegue de TechCorp AI Platform
# Uso: ./deploy.sh [entorno] [versión]

set -euo pipefail

# Colores para salida
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # Sin Color

# Valores por defecto
ENTORNO="${1:-staging}"
VERSION="${2:-latest}"
NAMESPACE="techcorp-${ENTORNO}"
IMAGE="ghcr.io/techcorp/ai-platform"
KUBECONFIG_FILE="${KUBECONFIG:-$HOME/.kube/config}"

# Funciones
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[ÉXITO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[ADVERTENCIA]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Verificando prerrequisitos..."
    
    # Verificar kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl no encontrado. Por favor instala kubectl."
        exit 1
    fi
    
    # Verificar conexión kubectl
    if ! kubectl cluster-info &> /dev/null; then
        log_error "No se puede conectar al clúster Kubernetes. Verifica tu kubeconfig."
        exit 1
    fi
    
    # Verificar que el namespace existe
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "El namespace $NAMESPACE no existe. Creando..."
        kubectl create namespace "$NAMESPACE"
    fi
    
    log_success "Verificación de prerrequisitos completada"
}

deploy() {
    local version="$1"
    local image="ghcr.io/techcorp/ai-platform:${version}"
    
    log_info "Desplegando versión ${version} en ${ENTORNO}..."
    
    # Actualizar imagen del deployment
    kubectl set image deployment/techcorp-ai-platform \
        api="ghcr.io/techcorp/ai-platform:${version}" \
        -n "${NAMESPACE}" \
        --record
    
    # Esperar rollout
    log_info "Esperando a que complete el rollout..."
    if kubectl rollout status deployment/techcorp-ai-platform \
        -n "${NAMESPACE}" \
        --timeout=300s; then
        log_success "Despliegue completado exitosamente"
    else
        log_error "El despliegue falló o expiró el tiempo"
        exit 1
    fi
    
    # Verificar despliegue
    verify_deployment
}

verify_deployment() {
    log_info "Verificando despliegue..."
    
    # Verificar que los pods estén listos
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
        log_success "Todos los ${total_pods} pods están listos"
    else
        log_error "Solo ${ready_pods}/${total_pods} pods están listos"
        return 1
    fi
    
    # Health check
    local service_ip
    service_ip=$(kubectl get service techcorp-ai-platform -n "${NAMESPACE}" -o jsonpath='{.spec.clusterIP}')
    
    if [[ -n "$service_ip" ]]; then
        log_info "IP del servicio: ${service_ip}"
        # Probar endpoint de salud
        if kubectl run -i --rm --restart=Never curl-test \
            --image=curlimages/curl:latest \
            -- curl -sf "http://techcorp-ai-platform.${NAMESPACE}.svc.cluster.local/health/live" &> /dev/null; then
            log_success "Health check pasó"
        else
            log_warning "Health check falló"
        fi
    fi
}

rollback() {
    log_warning "Haciendo rollback del despliegue..."
    kubectl rollout undo deployment/techcorp-ai-platform -n "${NAMESPACE}"
    kubectl rollout status deployment/techcorp-ai-platform -n "${NAMESPACE}" --timeout=120s
    log_success "Rollback completado"
}

# Script principal
main() {
    log_info "Iniciando despliegue en ${ENTORNO} (versión: ${VERSION})"
    
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
            log_error "Comando desconocido: ${2:-deploy}. Usa: deploy, rollback, o verify"
            exit 1
            ;;
    esac
}

# Ejecutar main
main "$@"