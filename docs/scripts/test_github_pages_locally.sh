#!/bin/bash
# 本地测试 GitHub Pages 双层部署
# 模拟 GitHub Pages 的部署结构并在本地预览

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║    本地测试 GitHub Pages 双层部署                                           ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 清理旧的测试目录
DEPLOY_ROOT="github-pages-test"
log_info "准备测试环境..."
rm -rf "$DEPLOY_ROOT"
mkdir -p "$DEPLOY_ROOT"

echo ""
log_info "测试目标："
echo "  1. 主网站: http://localhost:8080/"
echo "  2. API 文档: http://localhost:8080/documentation/"
echo ""

# ============================================================================
# 步骤 1: 准备主网站文件
# ============================================================================

log_info "步骤 1/4: 准备主网站文件..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 复制主网站文件（排除构建产物）
rsync -av --exclude='_out' --exclude='_build' --exclude='*.pyc' \
      --exclude='__pycache__' --exclude='.pytest_cache' \
      --exclude='scripts' \
      docs/ "$DEPLOY_ROOT/"

# 确保有 downloads_data.json
if [ ! -f "$DEPLOY_ROOT/downloads_data.json" ]; then
    log_warning "downloads_data.json 不存在，创建占位符"
    cat > "$DEPLOY_ROOT/downloads_data.json" << 'EOF'
{
  "generated_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "version_metadata": {},
  "download_links": {}
}
EOF
fi

# 创建文档链接说明
cat > "$DEPLOY_ROOT/DOCUMENTATION.md" << 'EOF'
# ACloudViewer Documentation

Full API documentation is available at: [/documentation/](/documentation/)

## Documentation Structure

- **Main Website**: Current page (project info, downloads, getting started)
- **API Documentation**: [/documentation/](/documentation/) (Python API, C++ API, Tutorials)

## Quick Links

- [Python API Reference](/documentation/python_api/)
- [C++ API Reference](/documentation/cpp_api/)
- [Tutorials](/documentation/tutorial/)
EOF

log_success "主网站文件准备完成"
echo "  - 主页: index.html"
echo "  - 下载数据: downloads_data.json"
echo "  - 文档链接: DOCUMENTATION.md"
echo ""

# ============================================================================
# 步骤 2: 构建 API 文档
# ============================================================================

log_info "步骤 2/4: 构建 API 文档..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 检查是否已有构建好的文档
if [ -d "docs/_out/html" ]; then
    log_info "发现已构建的文档: docs/_out/html"
    read -p "是否使用现有文档？(Y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        USE_EXISTING=true
    else
        USE_EXISTING=false
    fi
else
    USE_EXISTING=false
fi

if [ "$USE_EXISTING" = false ]; then
    log_info "构建文档（这可能需要几分钟）..."
    
    # 选择构建方法
    echo ""
    echo "选择构建方法："
    echo "  1. Docker 构建（推荐，完整构建）"
    echo "  2. 本地构建（需要已编译 Python 模块）"
    echo "  3. 简易构建（仅 Doxygen + Sphinx，无 Python 模块）"
    echo ""
    read -p "请选择 (1/2/3): " -n 1 -r BUILD_METHOD
    echo ""
    
    case $BUILD_METHOD in
        1)
            log_info "使用 Docker 构建..."
            if ! command -v docker &> /dev/null; then
                log_error "Docker 未安装！"
                exit 1
            fi
            
            # 构建 Docker 镜像
            docker build -t acloudviewer-ci:docs -f docker/Dockerfile.docs . || {
                log_error "Docker 构建失败！"
                exit 1
            }
            
            # 提取文档
            docker run -v "${PWD}:/opt/mount" --rm acloudviewer-ci:docs \
                bash -c "cp /root/ACloudViewer/acloudviewer-*-docs.tar.gz /opt/mount/ && \
                         chown $(id -u):$(id -g) /opt/mount/acloudviewer-*-docs.tar.gz"
            
            # 解压
            mkdir -p docs/_out/html
            tar -xzf acloudviewer-*-docs.tar.gz -C docs/_out/html/
            rm acloudviewer-*-docs.tar.gz
            ;;
        2)
            log_info "使用本地构建..."
            cd docs
            python3 make_docs.py --sphinx --doxygen --parallel
            cd ..
            ;;
        3)
            log_info "使用简易构建..."
            cd docs
            python3 make_docs.py --sphinx --doxygen
            cd ..
            ;;
        *)
            log_error "无效的选择！"
            exit 1
            ;;
    esac
fi

# 检查构建结果
if [ ! -d "docs/_out/html" ]; then
    log_error "文档构建失败！docs/_out/html 不存在"
    exit 1
fi

log_success "API 文档构建完成"
echo ""

# ============================================================================
# 步骤 3: 部署 API 文档到 documentation/ 子目录
# ============================================================================

log_info "步骤 3/4: 部署 API 文档到 documentation/ 子目录..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p "$DEPLOY_ROOT/documentation"
cp -r docs/_out/html/* "$DEPLOY_ROOT/documentation/"

log_success "API 文档部署完成"
echo ""

# ============================================================================
# 步骤 4: 验证部署结构
# ============================================================================

log_info "步骤 4/4: 验证部署结构..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

ERRORS=0

# 检查主网站文件
if [ ! -f "$DEPLOY_ROOT/index.html" ]; then
    log_error "缺少主网站: index.html"
    ERRORS=$((ERRORS + 1))
else
    log_success "主网站: index.html ✓"
fi

if [ ! -f "$DEPLOY_ROOT/downloads_data.json" ]; then
    log_warning "缺少下载数据: downloads_data.json"
else
    log_success "下载数据: downloads_data.json ✓"
fi

# 检查 API 文档
if [ ! -d "$DEPLOY_ROOT/documentation" ]; then
    log_error "缺少 API 文档目录: documentation/"
    ERRORS=$((ERRORS + 1))
else
    log_success "API 文档目录: documentation/ ✓"
fi

if [ ! -f "$DEPLOY_ROOT/documentation/index.html" ]; then
    log_error "缺少 API 文档主页: documentation/index.html"
    ERRORS=$((ERRORS + 1))
else
    log_success "API 文档主页: documentation/index.html ✓"
fi

# 检查 Python API
if [ -d "$DEPLOY_ROOT/documentation/python_api" ]; then
    PYTHON_API_FILES=$(find "$DEPLOY_ROOT/documentation/python_api" -name "*.html" | wc -l)
    log_success "Python API: $PYTHON_API_FILES 个 HTML 文件 ✓"
else
    log_warning "Python API 目录不存在"
fi

# 检查 C++ API
if [ -d "$DEPLOY_ROOT/documentation/cpp_api" ]; then
    CPP_API_FILES=$(find "$DEPLOY_ROOT/documentation/cpp_api" -name "*.html" | wc -l)
    log_success "C++ API: $CPP_API_FILES 个 HTML 文件 ✓"
else
    log_warning "C++ API 目录不存在"
fi

# 检查教程
if [ -d "$DEPLOY_ROOT/documentation/tutorial" ]; then
    TUTORIAL_FILES=$(find "$DEPLOY_ROOT/documentation/tutorial" -name "*.html" | wc -l)
    log_success "教程: $TUTORIAL_FILES 个 HTML 文件 ✓"
else
    log_warning "教程目录不存在"
fi

echo ""

if [ $ERRORS -gt 0 ]; then
    log_error "发现 $ERRORS 个错误！"
    exit 1
fi

# ============================================================================
# 统计信息
# ============================================================================

echo ""
log_info "部署统计："
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

TOTAL_FILES=$(find "$DEPLOY_ROOT" -type f | wc -l)
TOTAL_HTML=$(find "$DEPLOY_ROOT" -name "*.html" | wc -l)
TOTAL_SIZE=$(du -sh "$DEPLOY_ROOT" | cut -f1)

echo "  - 总文件数: $TOTAL_FILES"
echo "  - HTML 页面: $TOTAL_HTML"
echo "  - 总大小: $TOTAL_SIZE"
echo ""

# ============================================================================
# 启动本地服务器
# ============================================================================

log_success "部署结构验证通过！✅"
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║    准备启动本地服务器                                                        ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

log_info "部署结构："
echo ""
echo "  $DEPLOY_ROOT/"
echo "  ├── index.html                    # 主网站"
echo "  ├── downloads_data.json           # 下载数据"
echo "  ├── DOCUMENTATION.md              # 文档链接说明"
echo "  ├── [其他网站文件]"
echo "  └── documentation/                # API 文档子目录"
echo "      ├── index.html                # API 文档主页"
echo "      ├── python_api/               # Python API"
echo "      ├── cpp_api/                  # C++ API"
echo "      └── tutorial/                 # 教程"
echo ""

log_info "访问 URL："
echo ""
echo "  🌐 主网站:     http://localhost:8080/"
echo "  📚 API 文档:   http://localhost:8080/documentation/"
echo "  🐍 Python API: http://localhost:8080/documentation/python_api/"
echo "  ⚙️  C++ API:   http://localhost:8080/documentation/cpp_api/"
echo "  📖 教程:       http://localhost:8080/documentation/tutorial/"
echo ""

read -p "启动本地服务器？(Y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    log_info "启动服务器在 http://localhost:8080"
    log_warning "按 Ctrl+C 停止服务器"
    echo ""
    
    cd "$DEPLOY_ROOT"
    
    # 尝试使用 Python 3 的 http.server
    if command -v python3 &> /dev/null; then
        python3 -m http.server 8080
    elif command -v python &> /dev/null; then
        python -m http.server 8080
    else
        log_error "未找到 Python！请安装 Python 3"
        exit 1
    fi
fi
