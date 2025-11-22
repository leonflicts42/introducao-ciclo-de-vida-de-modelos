#!/bin/bash
# Script de demonstração do pipeline CI/CD
# Execute: bash demo.sh

set -e  # Para na primeira falha

echo "======================================================================"
echo "   DEMO: Pipeline CI/CD para Machine Learning"
echo "   Aula 06 - Automação com GitHub Actions e MLflow"
echo "======================================================================"
echo ""

# Cores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Função para print colorido
print_step() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Verificar que estamos no diretório correto
if [ ! -f "train.py" ]; then
    echo "Erro: Execute este script de dentro da pasta aula_06_cicd_automacao/"
    exit 1
fi

# ======================================================================
# PASSO 1: TESTES
# ======================================================================
print_step "PASSO 1: Executando Testes (Job 1 do Pipeline)"
print_info "Validando transformers e funções..."

python test_pipeline.py

if [ $? -eq 0 ]; then
    print_success "Todos os testes passaram!"
else
    echo "Erro: Testes falharam. Pipeline seria interrompido aqui."
    exit 1
fi

# ======================================================================
# PASSO 2: TREINAR BASELINE
# ======================================================================
print_step "PASSO 2: Treinando Modelo Baseline (Job 2 do Pipeline)"
print_info "Usando hiperparâmetros padrão..."

python train.py --model-type baseline --min-accuracy 0.75

print_success "Modelo baseline treinado e logado no MLflow!"

# ======================================================================
# PASSO 3: TREINAR OTIMIZADO
# ======================================================================
print_step "PASSO 3: Treinando Modelo Otimizado (Job 3 do Pipeline)"
print_info "Usando hiperparâmetros da Aula 03..."

python train.py --model-type optimized --min-accuracy 0.75

print_success "Modelo otimizado treinado e logado no MLflow!"

# ======================================================================
# PASSO 4: VISUALIZAR RESULTADOS
# ======================================================================
print_step "PASSO 4: Visualizar Resultados no MLflow"

echo ""
echo "Para visualizar os experimentos e comparar modelos:"
echo ""
echo -e "${GREEN}mlflow ui${NC}"
echo ""
echo "Depois acesse: http://localhost:5000"
echo ""
echo "No MLflow UI você poderá:"
echo "  • Comparar métricas lado a lado"
echo "  • Ver hiperparâmetros de cada run"
echo "  • Verificar versões no Model Registry"
echo "  • Checar qual modelo tem alias 'Production'"
echo ""

# ======================================================================
# RESUMO FINAL
# ======================================================================
print_step "RESUMO DO PIPELINE"

echo "✅ PASSO 1: Testes executados com sucesso"
echo "✅ PASSO 2: Modelo baseline treinado e logado"
echo "✅ PASSO 3: Modelo otimizado treinado e logado"
echo "📊 PASSO 4: Resultados disponíveis no MLflow"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${GREEN}Pipeline CI/CD executado com sucesso! 🎉${NC}"
echo ""
echo "Próximos passos:"
echo "  1. Execute 'mlflow ui' para visualizar resultados"
echo "  2. Modifique hiperparâmetros em train.py"
echo "  3. Execute novamente este script"
echo "  4. Compare as novas métricas no MLflow UI"
echo "  5. Faça commit e push para ver o pipeline no GitHub Actions!"
echo ""
echo "======================================================================"
