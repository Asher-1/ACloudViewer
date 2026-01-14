// ===== INTERNATIONALIZATION (i18n) SYSTEM =====
const translations = {
    en: {
        // Navigation
        'nav.home': 'Home',
        'nav.features': 'Features',
        'nav.download': 'Download',
        'nav.quickstart': 'Quick Start',
        'nav.gallery': 'Gallery',
        'nav.documentation': 'Documentation',
        'nav.donate': 'Support',
        
        // Hero Section
        'hero.subtitle': 'Modern 3D Data Processing System',
        'hero.description': 'Professional Point Cloud & Mesh Processing | Big Data Support | Cross-Platform Solution',
        'hero.download': 'Download Now',
        'hero.quickstart': 'Quick Start',
        'hero.opensource': 'Open Source',
        'hero.community': 'Active Community',
        'hero.crossplatform': 'Cross-Platform',
        
        // About Section
        'about.title': 'About ACloudViewer',
        'about.intro': 'ACloudViewer is an open-source 3D point cloud and triangular mesh processing software library. It supports rapid development of software for processing 3D data, highly based on CloudCompare, Open3D, ParaView and COLMAP, and integrates the PCL library.',
        'about.details': 'Originally designed to compare two 3D point clouds (such as those obtained by laser scanning) or the difference between point clouds and triangular meshes. It relies on an octree structure highly optimized for this specific use case, capable of handling massive point cloud data (typically over 10 million points, up to 120 million points with 2GB memory).',
        
        // Features Section
        'features.title': 'Core Features',
        'features.data_structure.title': '3D Data Structures',
        'features.data_structure.desc': 'Powerful 3D data structures and processing algorithms, supporting point clouds, meshes and various geometries',
        'features.reconstruction.title': 'Scene Reconstruction',
        'features.reconstruction.desc': 'COLMAP-based scene reconstruction system, supporting complete workflow from images to 3D models',
        'features.registration.title': 'Surface Alignment',
        'features.registration.desc': 'High-precision point cloud registration algorithms, including ICP, RANSAC and other methods',
        'features.visualization.title': '3D Visualization',
        'features.visualization.desc': 'High-performance rendering engine based on VTK and OpenGL, supporting PBR physical rendering',
        'features.ml.title': 'Machine Learning',
        'features.ml.desc': 'Integrated with PyTorch and TensorFlow, supporting 3D deep learning applications',
        'features.gpu.title': 'GPU Acceleration',
        'features.gpu.desc': 'GPU acceleration for core 3D operations, supporting CUDA 12.x',
        'features.api.title': 'C++ & Python',
        'features.api.desc': 'Provides C++ and Python dual-language API, flexible and easy to use',
        'features.plugins.title': 'Plugin System',
        'features.plugins.desc': 'Rich plugin ecosystem, supporting custom feature extensions',
        
        // Download Section
        'download.title': 'Download ACloudViewer',
        'download.subtitle': 'Choose the version for your system',
        'download.intro_text': 'All current and past release downloads are available on',
        'download.github_releases': 'GitHub releases',
        'download.version_label': 'ACloudViewer version',
        'download.os_label': 'OS',
        'download.linux_distro_label': 'Linux Distribution',
        'download.package_label': 'Package',
        'download.python_label': 'Python',
        'download.cuda_label': 'CUDA',
        'download.arch_label': 'Architecture',
        'download.cpu_only': 'CPU Only',
        'download.result_label': 'Link:',
        'download.select_options': 'Select options above to see available downloads',
        'download.loading': 'Loading versions...',
        'download.file_size': 'Size',
        'download.not_available': 'Not available. Please try a different combination of options.',
        'download.beta_version': '🚀 Beta Version: Devel packages from the main branch',
        'download.release_date': 'Release Date',
        'download.commit': 'Commit',
        'download.beta_desc': 'Latest test version with newest features and improvements',
        'download.stable_version': '✅ Stable Version',
        'download.stable_desc': 'Tested stable version, recommended for production',
        'download.windows': 'Windows',
        'download.macos': 'macOS',
        'download.linux': 'Linux',
        'download.download_btn': 'Download',
        'download.python_package': 'Python Package',
        'download.install_pip': 'Install via pip',
        'download.python_install_title': 'Python Installation',
        'download.python_step1_title': 'Download Wheel File',
        'download.python_step1_desc': 'Download the .whl file for your system and Python version from',
        'download.python_step1_note': '💡 Due to file size exceeding PyPI limits, manual download is required',
        'download.python_step2_title': 'Install Wheel File',
        'download.python_step2_example': 'Example:',
        'download.python_support_note': 'Supports Python 3.10-3.12 | Ubuntu 20.04+, macOS 10.15+, Windows 10+ (64-bit)',
        
        // Quick Start Section
        'quickstart.title': 'Quick Start',
        'quickstart.subtitle': 'Get started with ACloudViewer in 5 minutes',
        'quickstart.windows': 'Windows',
        'quickstart.macos': 'macOS',
        'quickstart.linux': 'Linux',
        'quickstart.python': 'Python',
        'quickstart.step1.title': 'Download and Install',
        'quickstart.step1.desc_windows': 'Double-click the downloaded .exe file and follow the installation wizard',
        'quickstart.step1.desc_macos': 'Open the downloaded .dmg file and drag the application to the Applications folder',
        'quickstart.step1.desc_linux': 'Make the .run file executable and run it',
        'quickstart.step1.desc_python': 'Install via pip (Python 3.10-3.12)',
        'quickstart.step2.title': 'Launch Application',
        'quickstart.step2.desc_gui': 'Launch ACloudViewer from the Start Menu or Desktop shortcut',
        'quickstart.step2.desc_python': 'Import ACloudViewer in Python',
        'quickstart.step3.title': 'Load Data',
        'quickstart.step3.desc_gui': 'Click "Open" button or drag and drop your 3D data files',
        'quickstart.step3.desc_python': 'Load your first point cloud',
        'quickstart.view_docs': 'View Documentation',
        'quickstart.python_tab': 'Python',
        'quickstart.cpp_tab': 'C++',
        'quickstart.gui_tab': 'GUI App',
        'quickstart.python_intro_title': 'Python Quick Start',
        'quickstart.python_step1_title': 'Download & Install',
        'quickstart.python_step1_desc': 'Download the corresponding .whl file from',
        'quickstart.python_step1_note': '💡 Due to large file size, direct PyPI installation is not supported',
        'quickstart.python_step2_title': 'Verify Installation',
        'quickstart.python_step3_title': 'Run Example',
        'quickstart.python_example_comment1': '# Create sphere mesh',
        'quickstart.python_example_comment2': '# Visualize',
        
        // Gallery Section
        'gallery.title': 'Application Showcase',
        'gallery.subtitle': 'Explore ACloudViewer\'s powerful applications in different fields',
        'gallery.ui': 'User Interface',
        'gallery.reconstruction': '3D Reconstruction',
        'gallery.annotation': 'Semantic Annotation',
        'gallery.ml': 'Machine Learning',
        'gallery.visualization': 'Visualization',
        'gallery.cloudviewer_app.title': 'CloudViewer App',
        'gallery.cloudviewer_app.desc': 'Lightweight point cloud viewer',
        'gallery.architecture.title': 'System Architecture',
        'gallery.architecture.desc': 'Modular design, complete abstraction from bottom layer to application layer',
        'gallery.reconstruction_img.title': 'Scene Reconstruction',
        'gallery.reconstruction_img.desc': 'Complete 3D reconstruction workflow based on COLMAP',
        'gallery.realtime.title': 'Real-time 3D Reconstruction',
        'gallery.realtime.desc': 'GPU-accelerated real-time point cloud reconstruction and fusion',
        'gallery.icp.title': 'ICP Point Cloud Registration',
        'gallery.icp.desc': 'High-performance iterative closest point algorithm, supporting multi-scale registration',
        'gallery.main_ui.title': 'Main Interface',
        'gallery.main_ui.desc': 'Professional 3D data processing and visualization interface',
        'gallery.semantic.title': 'Semantic Annotation Tool',
        'gallery.semantic.desc': 'Intelligent 3D semantic segmentation and annotation',
        'gallery.scene_cloud.title': 'Large-scale Scene Annotation',
        'gallery.scene_cloud.desc': 'Semantic annotation for massive point cloud data, supporting rendering of hundreds of millions of points',
        'gallery.ml_vis.title': 'Machine Learning Visualization',
        'gallery.ml_vis.desc': 'Real-time visualization of 3D machine learning model training and inference',
        'gallery.ml_pred.title': 'Model Prediction Visualization',
        'gallery.ml_pred.desc': 'Real-time visualization of 3D deep learning model inference results',
        'gallery.jupyter.title': 'Jupyter Integration',
        'gallery.jupyter.desc': 'Interactive 3D data visualization in Jupyter Notebook',
        'gallery.pbr.title': 'PBR Rendering',
        'gallery.pbr.desc': 'Physics-based rendering, supporting materials, lighting and shadows',
        
        // Resources Section
        'resources.title': 'Resources & Documentation',
        'resources.docs.title': 'Documentation',
        'resources.docs.desc': 'Complete API documentation and user guides',
        'resources.examples.title': 'Examples',
        'resources.examples.desc': 'Rich code examples and tutorials',
        'resources.community.title': 'Community',
        'resources.community.desc': 'Join our active community and discuss',
        'resources.github.title': 'Source Code',
        'resources.github.desc': 'View source code and contribute',
        
        // Footer
        'footer.about': 'About',
        'footer.about_desc': 'Professional 3D data processing system',
        'footer.quick_links': 'Quick Links',
        'footer.documentation': 'Documentation',
        'footer.releases': 'Releases',
        'footer.contribute': 'Contribute',
        'footer.issues': 'Issues',
        'footer.community': 'Community',
        'footer.github': 'GitHub',
        'footer.discussions': 'Discussions',
        'footer.social': 'Social Media',
        'footer.license': 'License',
        'footer.copyright': '© 2025 ACloudViewer. All rights reserved. | Licensed under GPL-2.0 and MIT',
    },
    zh: {
        // 导航
        'nav.home': '首页',
        'nav.features': '特性',
        'nav.download': '下载',
        'nav.quickstart': '快速开始',
        'nav.gallery': '展示',
        'nav.documentation': '文档',
        'nav.donate': '支持',
        
        // 首页
        'hero.subtitle': '现代化的3D数据处理系统',
        'hero.description': '专业的点云和三角网格处理软件 | 支持海量数据 | 跨平台解决方案',
        'hero.download': '立即下载',
        'hero.quickstart': '快速开始',
        'hero.opensource': '开源',
        'hero.community': '活跃社区',
        'hero.crossplatform': '跨平台',
        
        // 关于部分
        'about.title': '关于 ACloudViewer',
        'about.intro': 'ACloudViewer 是一个开源的3D点云和三角网格处理软件库。它支持快速开发处理3D数据的软件，高度基于 CloudCompare、Open3D、ParaView 和 COLMAP，并集成了PCL库。',
        'about.details': '最初设计用于比较两个3D点云（如激光扫描获得的点云）或点云与三角网格之间的差异。它依赖于针对此特定用例高度优化的八叉树结构，能够处理海量点云数据（通常超过1000万点，在2GB内存下最多可达1.2亿点）。',
        
        // 特性部分
        'features.title': '核心特性',
        'features.data_structure.title': '3D数据结构',
        'features.data_structure.desc': '强大的3D数据结构和处理算法，支持点云、网格和各种几何体',
        'features.reconstruction.title': '场景重建',
        'features.reconstruction.desc': '基于COLMAP的场景重建系统，支持从图像到3D模型的完整流程',
        'features.registration.title': '表面对齐',
        'features.registration.desc': '高精度的点云配准算法，包括ICP、RANSAC等多种方法',
        'features.visualization.title': '3D可视化',
        'features.visualization.desc': '基于VTK和OpenGL的高性能渲染引擎，支持PBR物理渲染',
        'features.ml.title': '机器学习',
        'features.ml.desc': '集成PyTorch和TensorFlow，支持3D深度学习应用',
        'features.gpu.title': 'GPU加速',
        'features.gpu.desc': '核心3D操作的GPU加速，支持CUDA 12.x',
        'features.api.title': 'C++ & Python',
        'features.api.desc': '提供C++和Python双语言API，灵活易用',
        'features.plugins.title': '插件系统',
        'features.plugins.desc': '丰富的插件生态系统，支持自定义功能扩展',
        
        // 下载部分
        'download.title': '下载 ACloudViewer',
        'download.subtitle': '选择适合您系统的版本',
        'download.intro_text': '所有当前和历史版本均可在',
        'download.github_releases': 'GitHub releases',
        'download.version_label': 'ACloudViewer 版本',
        'download.os_label': '操作系统',
        'download.linux_distro_label': 'Linux 发行版',
        'download.package_label': '包类型',
        'download.python_label': 'Python 版本',
        'download.cuda_label': 'CUDA 支持',
        'download.arch_label': '架构',
        'download.cpu_only': '仅CPU',
        'download.result_label': '下载链接:',
        'download.select_options': '请在上方选择选项以查看可用下载',
        'download.loading': '正在加载版本...',
        'download.file_size': '大小',
        'download.not_available': '该组合暂不可用，请尝试其他选项组合。',
        'download.beta_version': '🚀 Beta 版本: Devel packages from the main branch',
        'download.release_date': '发布日期',
        'download.commit': 'Commit',
        'download.beta_desc': '最新的测试版本，包含最新功能和改进',
        'download.stable_version': '✅ 稳定版本',
        'download.stable_desc': '经过测试的稳定版本，推荐用于生产环境',
        'download.windows': 'Windows',
        'download.macos': 'macOS',
        'download.linux': 'Linux',
        'download.download_btn': '下载',
        'download.python_package': 'Python 包',
        'download.install_pip': '通过 pip 安装',
        'download.python_install_title': 'Python 安装',
        'download.python_step1_title': '下载 Wheel 文件',
        'download.python_step1_desc': '从 GitHub Releases 下载对应您系统和 Python 版本的 .whl 文件',
        'download.python_step1_note': '💡 由于文件体积超过 PyPI 限制，需手动下载安装',
        'download.python_step2_title': '安装 Wheel 文件',
        'download.python_step2_example': '示例：',
        'download.python_support_note': '支持 Python 3.10-3.12 | Ubuntu 20.04+, macOS 10.15+, Windows 10+ (64-bit)',
        
        // 快速开始部分
        'quickstart.title': '快速开始',
        'quickstart.subtitle': '5分钟上手 ACloudViewer',
        'quickstart.windows': 'Windows',
        'quickstart.macos': 'macOS',
        'quickstart.linux': 'Linux',
        'quickstart.python': 'Python',
        'quickstart.step1.title': '下载和安装',
        'quickstart.step1.desc_windows': '双击下载的 .exe 文件，按照安装向导完成安装',
        'quickstart.step1.desc_macos': '打开下载的 .dmg 文件，将应用程序拖拽到应用程序文件夹',
        'quickstart.step1.desc_linux': '给 .run 文件添加执行权限并运行',
        'quickstart.step1.desc_python': '通过 pip 安装（Python 3.10-3.12）',
        'quickstart.step2.title': '启动应用',
        'quickstart.step2.desc_gui': '从开始菜单或桌面快捷方式启动 ACloudViewer',
        'quickstart.step2.desc_python': '在 Python 中导入 ACloudViewer',
        'quickstart.step3.title': '加载数据',
        'quickstart.step3.desc_gui': '点击"打开"按钮或拖拽3D数据文件',
        'quickstart.step3.desc_python': '加载你的第一个点云',
        'quickstart.view_docs': '查看文档',
        'quickstart.python_tab': 'Python',
        'quickstart.cpp_tab': 'C++',
        'quickstart.gui_tab': 'GUI应用',
        'quickstart.python_intro_title': 'Python 快速入门',
        'quickstart.python_step1_title': '下载并安装',
        'quickstart.python_step1_desc': '从 GitHub Releases 下载对应的 .whl 文件，然后安装：',
        'quickstart.python_step1_note': '💡 文件体积较大，暂不支持 PyPI 直接安装',
        'quickstart.python_step2_title': '验证安装',
        'quickstart.python_step3_title': '运行示例',
        'quickstart.python_example_comment1': '# 创建球体网格',
        'quickstart.python_example_comment2': '# 可视化',
        
        // 图库部分
        'gallery.title': '应用展示',
        'gallery.subtitle': '探索 ACloudViewer 在不同领域的强大应用',
        'gallery.ui': '用户界面',
        'gallery.reconstruction': '3D重建',
        'gallery.annotation': '语义标注',
        'gallery.ml': '机器学习',
        'gallery.visualization': '可视化',
        'gallery.cloudviewer_app.title': 'CloudViewer 应用',
        'gallery.cloudviewer_app.desc': '轻量级的点云查看器',
        'gallery.architecture.title': '系统架构',
        'gallery.architecture.desc': '模块化设计，从底层到应用层的完整抽象',
        'gallery.reconstruction_img.title': '场景重建',
        'gallery.reconstruction_img.desc': '基于COLMAP的完整3D重建流程',
        'gallery.realtime.title': '实时3D重建',
        'gallery.realtime.desc': 'GPU加速的实时点云重建与融合',
        'gallery.icp.title': 'ICP点云配准',
        'gallery.icp.desc': '高性能的迭代最近点算法，支持多尺度配准',
        'gallery.main_ui.title': '主界面',
        'gallery.main_ui.desc': '专业的3D数据处理与可视化界面',
        'gallery.semantic.title': '语义标注工具',
        'gallery.semantic.desc': '智能的3D语义分割和标注功能',
        'gallery.scene_cloud.title': '大规模场景标注',
        'gallery.scene_cloud.desc': '处理海量点云数据的语义标注，支持上亿点渲染',
        'gallery.ml_vis.title': '机器学习可视化',
        'gallery.ml_vis.desc': '实时可视化3D机器学习模型训练和推理过程',
        'gallery.ml_pred.title': '模型预测可视化',
        'gallery.ml_pred.desc': '3D深度学习模型推理结果的实时可视化展示',
        'gallery.jupyter.title': 'Jupyter 集成',
        'gallery.jupyter.desc': '在Jupyter Notebook中交互式可视化3D数据',
        'gallery.pbr.title': 'PBR渲染',
        'gallery.pbr.desc': '基于物理的渲染，支持材质、光照和阴影',
        
        // 资源部分
        'resources.title': '资源与文档',
        'resources.docs.title': '文档',
        'resources.docs.desc': '完整的API文档和用户指南',
        'resources.examples.title': '示例',
        'resources.examples.desc': '丰富的代码示例和教程',
        'resources.community.title': '社区',
        'resources.community.desc': '加入我们活跃的社区，交流讨论',
        'resources.github.title': '源代码',
        'resources.github.desc': '查看源代码并贡献',
        
        // 页脚
        'footer.about': '关于',
        'footer.about_desc': '专业的3D数据处理系统',
        'footer.quick_links': '快速链接',
        'footer.documentation': '文档',
        'footer.releases': '发布版本',
        'footer.contribute': '贡献',
        'footer.issues': '问题反馈',
        'footer.community': '社区',
        'footer.github': 'GitHub',
        'footer.discussions': '讨论',
        'footer.social': '社交媒体',
        'footer.license': '许可证',
        'footer.copyright': '© 2025 ACloudViewer. 保留所有权利 | 采用 GPL-2.0 和 MIT 双重许可',
    }
};

// Current language (default to English)
let currentLang = localStorage.getItem('preferredLanguage') || 'en';

// Function to translate the page
function translatePage(lang) {
    currentLang = lang;
    localStorage.setItem('preferredLanguage', lang);
    
    // Update HTML lang attribute
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    
    // Update meta tags
    const descMeta = document.querySelector('meta[name="description"]');
    const keywordsMeta = document.querySelector('meta[name="keywords"]');
    const title = document.querySelector('title');
    
    if (lang === 'zh') {
        if (descMeta) descMeta.content = 'ACloudViewer - 专业的3D点云和网格处理软件，支持Windows、Linux和macOS';
        if (keywordsMeta) keywordsMeta.content = '点云处理,3D重建,点云可视化,开源软件,CloudCompare,Open3D';
        if (title) title.textContent = 'ACloudViewer - 现代化3D数据处理系统';
    } else {
        if (descMeta) descMeta.content = 'ACloudViewer - Professional 3D point cloud and mesh processing software, supporting Windows, Linux and macOS';
        if (keywordsMeta) keywordsMeta.content = 'point cloud processing,3D reconstruction,point cloud visualization,open source,CloudCompare,Open3D';
        if (title) title.textContent = 'ACloudViewer - Modern 3D Data Processing System';
    }
    
    // Update current language display
    const currentLangSpan = document.getElementById('currentLang');
    if (currentLangSpan) {
        currentLangSpan.textContent = lang === 'zh' ? '中文' : 'EN';
    }
    
    // Update all elements with data-i18n attribute
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        if (translations[lang] && translations[lang][key]) {
            element.textContent = translations[lang][key];
        }
    });
    
    // Update download buttons (dynamically update button text)
    document.querySelectorAll('.btn-download-small, .btn-download').forEach(btn => {
        const iconHtml = '<i class="fas fa-download"></i>';
        const buttonText = lang === 'zh' ? '下载' : 'Download';
        if (btn.innerHTML.includes('fa-download')) {
            btn.innerHTML = `${iconHtml} ${buttonText}`;
        }
    });
    
    // Update copy buttons
    document.querySelectorAll('.copy-btn').forEach(btn => {
        const originalText = btn.getAttribute('data-original-text');
        if (!originalText) {
            btn.setAttribute('data-original-text', btn.innerHTML);
        }
        if (!btn.innerHTML.includes('fa-check')) {
            btn.innerHTML = lang === 'zh' ? '<i class="fas fa-copy"></i> 复制' : '<i class="fas fa-copy"></i> Copy';
        }
    });
    
    // Update "View Documentation" links
    document.querySelectorAll('a[href*="docs"]').forEach(link => {
        if (link.textContent.includes('查看文档') || link.textContent.includes('View Documentation')) {
            link.textContent = lang === 'zh' ? '查看文档' : 'View Documentation';
        }
    });
    
    // Update console easter egg
    if (lang === 'zh') {
        console.log('%c🎉 欢迎使用 ACloudViewer！', 'font-size: 16px; color: #FFC107;');
    } else {
        console.log('%c🎉 Welcome to ACloudViewer!', 'font-size: 16px; color: #FFC107;');
    }
}

// Language switcher event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Initialize with saved or default language
    translatePage(currentLang);
    
    // Language dropdown options
    const langOptions = document.querySelectorAll('.lang-option');
    langOptions.forEach(option => {
        option.addEventListener('click', () => {
            const lang = option.getAttribute('data-lang');
            translatePage(lang);
            
            // Update active state
            langOptions.forEach(opt => opt.classList.remove('active'));
            option.classList.add('active');
        });
        
        // Set initial active state
        if (option.getAttribute('data-lang') === currentLang) {
            option.classList.add('active');
        }
    });
});

// Mobile Menu Toggle
const hamburger = document.querySelector('.hamburger');
const navMenu = document.querySelector('.nav-menu');

hamburger.addEventListener('click', () => {
    navMenu.classList.toggle('active');
    hamburger.classList.toggle('active');
});

// Close mobile menu when clicking on a link
document.querySelectorAll('.nav-menu a').forEach(link => {
    link.addEventListener('click', () => {
        navMenu.classList.remove('active');
        hamburger.classList.remove('active');
    });
});

// Navbar scroll effect
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
    if (window.scrollY > 100) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
});

// Tabs functionality
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const tabName = btn.getAttribute('data-tab');
        
        // Remove active class from all tabs and contents
        tabBtns.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));
        
        // Add active class to clicked tab and corresponding content
        btn.classList.add('active');
        document.getElementById(`${tabName}-tab`).classList.add('active');
    });
});

// Version tabs functionality
const versionTabs = document.querySelectorAll('.version-tab');
const versionContents = document.querySelectorAll('.version-content');

versionTabs.forEach(tab => {
    tab.addEventListener('click', () => {
        const version = tab.getAttribute('data-version');
        
        // Remove active class from all version tabs and contents
        versionTabs.forEach(t => t.classList.remove('active'));
        versionContents.forEach(c => c.classList.remove('active'));
        
        // Add active class to clicked tab and corresponding content
        tab.classList.add('active');
        const content = document.getElementById(`version-${version}`);
        if (content) {
            content.classList.add('active');
        }
    });
});

// Gallery tabs functionality
const galleryTabs = document.querySelectorAll('.gallery-tab');
const galleryContents = document.querySelectorAll('.gallery-content');

galleryTabs.forEach(tab => {
    tab.addEventListener('click', () => {
        const gallery = tab.getAttribute('data-gallery');
        
        // Remove active class from all gallery tabs and contents
        galleryTabs.forEach(t => t.classList.remove('active'));
        galleryContents.forEach(c => c.classList.remove('active'));
        
        // Add active class to clicked tab and corresponding content
        tab.classList.add('active');
        const content = document.getElementById(`gallery-${gallery}`);
        if (content) {
            content.classList.add('active');
        }
    });
});

// Copy to clipboard functionality
document.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const text = btn.getAttribute('data-clipboard');
        
        // Create temporary textarea
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        
        // Select and copy
        textarea.select();
        document.execCommand('copy');
        
        // Remove textarea
        document.body.removeChild(textarea);
        
        // Show feedback
        const originalHTML = btn.innerHTML;
        btn.innerHTML = '<i class="fas fa-check"></i>';
        setTimeout(() => {
            btn.innerHTML = originalHTML;
        }, 2000);
    });
});

// Back to top button
const backToTopBtn = document.getElementById('backToTop');

window.addEventListener('scroll', () => {
    if (window.scrollY > 500) {
        backToTopBtn.classList.add('visible');
    } else {
        backToTopBtn.classList.remove('visible');
    }
});

backToTopBtn.addEventListener('click', () => {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
});

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        const href = this.getAttribute('href');
        if (href !== '#' && document.querySelector(href)) {
            e.preventDefault();
            const target = document.querySelector(href);
            const navbarHeight = navbar.offsetHeight;
            const targetPosition = target.offsetTop - navbarHeight;
            
            window.scrollTo({
                top: targetPosition,
                behavior: 'smooth'
            });
        }
    });
});

// Intersection Observer for animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe elements for animation
document.querySelectorAll('.feature-card, .download-card, .resource-card, .gallery-item').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(30px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(el);
});

// Parallax effect for hero section
window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const heroContent = document.querySelector('.hero-content');
    if (heroContent) {
        heroContent.style.transform = `translateY(${scrolled * 0.5}px)`;
        heroContent.style.opacity = 1 - (scrolled * 0.002);
    }
});

// Dynamic year for footer
const yearSpan = document.querySelector('.footer-bottom p');
if (yearSpan && !yearSpan.textContent.includes('2025')) {
    const currentYear = new Date().getFullYear();
    if (currentYear > 2025) {
        yearSpan.textContent = yearSpan.textContent.replace('2025', `2025-${currentYear}`);
    }
}

// Gallery lightbox effect (simple version)
document.querySelectorAll('.gallery-item').forEach(item => {
    item.addEventListener('click', () => {
        const img = item.querySelector('img');
        const lightbox = document.createElement('div');
        lightbox.className = 'lightbox';
        lightbox.innerHTML = `
            <div class="lightbox-content">
                <span class="lightbox-close">&times;</span>
                <img src="${img.src}" alt="${img.alt}">
            </div>
        `;
        
        // Add lightbox styles
        const style = document.createElement('style');
        style.textContent = `
            .lightbox {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0,0,0,0.9);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 10000;
                animation: fadeIn 0.3s;
            }
            .lightbox-content {
                position: relative;
                max-width: 90%;
                max-height: 90%;
            }
            .lightbox-content img {
                max-width: 100%;
                max-height: 90vh;
                object-fit: contain;
            }
            .lightbox-close {
                position: absolute;
                top: -40px;
                right: 0;
                color: white;
                font-size: 40px;
                cursor: pointer;
                transition: transform 0.3s;
            }
            .lightbox-close:hover {
                transform: scale(1.2);
            }
        `;
        
        document.head.appendChild(style);
        document.body.appendChild(lightbox);
        
        // Close lightbox
        const closeBtn = lightbox.querySelector('.lightbox-close');
        closeBtn.addEventListener('click', () => {
            lightbox.remove();
            style.remove();
        });
        
        lightbox.addEventListener('click', (e) => {
            if (e.target === lightbox) {
                lightbox.remove();
                style.remove();
            }
        });
    });
});

// Add loading animation
window.addEventListener('load', () => {
    document.body.style.opacity = '0';
    setTimeout(() => {
        document.body.style.transition = 'opacity 0.5s';
        document.body.style.opacity = '1';
    }, 100);
});

// Stats counter animation
function animateCounter(element, target, duration = 2000) {
    let current = 0;
    const increment = target / (duration / 16);
    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            element.textContent = Math.round(target);
            clearInterval(timer);
        } else {
            element.textContent = Math.round(current);
        }
    }, 16);
}

// Version badge pulse animation
const versionBadge = document.querySelector('.version-badge');
if (versionBadge) {
    setInterval(() => {
        versionBadge.style.transform = 'scale(1.05)';
        setTimeout(() => {
            versionBadge.style.transform = 'scale(1)';
        }, 200);
    }, 3000);
}

// Enhanced scroll reveal animation
const revealElements = document.querySelectorAll('.feature-card, .download-card, .step, .resource-card');
const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry, index) => {
        if (entry.isIntersecting) {
            setTimeout(() => {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }, index * 100);
            revealObserver.unobserve(entry.target);
        }
    });
}, {
    threshold: 0.1,
    rootMargin: '0px'
});

revealElements.forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(50px)';
    el.style.transition = 'all 0.6s ease';
    revealObserver.observe(el);
});

// Download button tracking and error handling
document.querySelectorAll('.btn-download, .btn-download-small').forEach(btn => {
    btn.addEventListener('click', (e) => {
        const href = btn.getAttribute('href');
        
        // Track download
        try {
            const platform = btn.closest('.download-card').querySelector('h4, h3')?.textContent || 'Unknown';
            const version = document.querySelector('.version-tab.active')?.textContent || 'Latest';
            console.log(`Download initiated: ${version} - ${platform}`);
            // Add your analytics code here
        } catch (error) {
            console.error('Error tracking download:', error);
        }

        // Handle 404 errors gracefully
        if (href && !href.includes('releases/latest')) {
            fetch(href, { method: 'HEAD' })
                .catch(() => {
                    console.warn('Download link may not be available:', href);
                });
        }
    });
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Press 'H' to go home
    if (e.key === 'h' || e.key === 'H') {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
    // Press 'Escape' to close mobile menu
    if (e.key === 'Escape') {
        navMenu.classList.remove('active');
        hamburger.classList.remove('active');
    }
});

// Add subtle parallax to feature cards
document.querySelectorAll('.feature-card').forEach(card => {
    card.addEventListener('mousemove', (e) => {
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        
        const rotateX = (y - centerY) / 20;
        const rotateY = (centerX - x) / 20;
        
        card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-10px)`;
    });
    
    card.addEventListener('mouseleave', () => {
        card.style.transform = '';
    });
});

// Preload images
const imageUrls = [
    'images/Annotaion.png',
    'images/SemanticAnnotation.png',
    'images/Reconstruction.png',
    'images/CloudViewerApp.png',
    'images/real-time-3D-Reconstruction.png',
    'images/SenceCloud.png',
    'images/ICP-registration.png',
    'images/AbstractionLayers.png',
    'images/jupyter_visualizer.png',
    'images/gifs/getting_started_ml_visualizer.gif'
];

imageUrls.forEach(url => {
    const img = new Image();
    img.src = url;
});

// Console easter egg
console.log('%cACloudViewer', 'font-size: 40px; font-weight: bold; color: #2196F3;');
console.log('%c🎉 欢迎使用 ACloudViewer！', 'font-size: 16px; color: #FFC107;');
console.log('%c如果你对项目感兴趣，欢迎访问: https://github.com/Asher-1/ACloudViewer', 'font-size: 14px;');

// Performance monitoring
if ('performance' in window) {
    window.addEventListener('load', () => {
        setTimeout(() => {
            const perfData = window.performance.timing;
            const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;
            console.log(`页面加载时间: ${pageLoadTime}ms`);
        }, 0);
    });
}

// ===== DOWNLOAD SELECTOR SYSTEM =====

let downloadsData = null;
let versionMetadata = [];
let currentSelection = {
    version: null,
    os: 'windows',
    package: 'app',
    linuxVersion: 'ubuntu20.04',
    python: null,
    cuda: 'cpu',
    arch: 'amd64'
};

// DOM elements (will be initialized on DOM ready)
let versionSelector = null;
let osSelector = null;
let linuxVersionRow = null;
let linuxVersionSelector = null;
let packageSelector = null;
let pythonRow = null;
let pythonSelector = null;
let cudaRow = null;
let cudaSelector = null;
let archRow = null;
let archSelector = null;
let downloadOutput = null;

// Load downloads data from downloads_data.json
async function loadDownloadsData() {
    console.log('📡 Loading downloads data from downloads_data.json...');
    
    try {
        const response = await fetch('downloads_data.json');
        console.log(`📥 Response status: ${response.status}`);
        
        if (!response.ok) {
            throw new Error(`Failed to load downloads_data.json (status: ${response.status})`);
        }
        
        const data = await response.json();
        console.log('✅ Data loaded successfully!');
        console.log('   Generated at:', data.generated_at);
        console.log('   Versions:', data.version_metadata.length);
        
        downloadsData = data.download_data;
        versionMetadata = data.version_metadata;
        
        console.log('📊 Version metadata:', versionMetadata);
        
        initializeVersionSelector();
        initializeSelectors();
        updateAvailableOptions();
        
    } catch (error) {
        console.error('❌ Error loading downloads data:', error);
        const versionSelector = document.getElementById('version-selector');
        if (versionSelector) {
            versionSelector.innerHTML = `<span class="result-message" style="color: #d32f2f;">Failed to load downloads data: ${error.message}<br><small>Please refresh the page. If the problem persists, visit <a href="https://github.com/Asher-1/ACloudViewer/releases" target="_blank">GitHub Releases</a> directly.</small></span>`;
        }
    }
}

function initializeVersionSelector() {
    versionSelector.innerHTML = '';
    
    versionMetadata.forEach((version, index) => {
        const btn = document.createElement('button');
        btn.className = 'selector-btn';
        btn.dataset.value = version.value;
        btn.dataset.pythonVersions = version.python_versions.join(',');
        btn.dataset.ubuntuVersions = version.ubuntu_versions.join(',');
        btn.textContent = version.display_name;
        
        if (version.is_default || index === 0) {
            btn.classList.add('active');
            currentSelection.version = version.value;
        }
        
        btn.addEventListener('click', () => {
            versionSelector.querySelectorAll('.selector-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentSelection.version = btn.dataset.value;
            updatePythonVersions(btn);
            updateUbuntuVersions(btn);
            updateAvailableOptions();
        });
        
        versionSelector.appendChild(btn);
    });
    
    // Initialize with first version
    const firstBtn = versionSelector.querySelector('.selector-btn.active');
    if (firstBtn) {
        updatePythonVersions(firstBtn);
        updateUbuntuVersions(firstBtn);
    }
}

function initializeSelectors() {
    // OS selector
    osSelector.querySelectorAll('.selector-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            osSelector.querySelectorAll('.selector-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentSelection.os = btn.dataset.value;
            
            // Show/hide Linux version selector
            if (btn.dataset.value === 'linux') {
                linuxVersionRow.style.display = 'flex';
                const currentVersionBtn = versionSelector.querySelector('.selector-btn.active');
                if (currentVersionBtn) {
                    updateUbuntuVersions(currentVersionBtn);
                }
            } else {
                linuxVersionRow.style.display = 'none';
            }
            
            updateAvailableOptions();
            updateDownloadLink();
        });
    });
    
    // Linux version selector
    linuxVersionSelector.querySelectorAll('.selector-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            linuxVersionSelector.querySelectorAll('.selector-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentSelection.linuxVersion = btn.dataset.value;
            updateAvailableOptions();
        });
    });
    
    // Package selector
    packageSelector.querySelectorAll('.selector-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            packageSelector.querySelectorAll('.selector-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentSelection.package = btn.dataset.value;
            
            // Show/hide Python selector
            if (btn.dataset.value === 'wheel') {
                pythonRow.style.display = 'flex';
                
                // Update available Python versions
                const currentVersionBtn = versionSelector.querySelector('.selector-btn.active');
                if (currentVersionBtn) {
                    updatePythonVersions(currentVersionBtn);
                }
                
                if (!currentSelection.python) {
                    const firstAvailable = pythonSelector.querySelector('.selector-btn:not([disabled])');
                    if (firstAvailable) {
                        firstAvailable.click();
                    }
                }
            } else {
                pythonRow.style.display = 'none';
                currentSelection.python = null;
            }
            
            updateAvailableOptions();
        });
    });
    
    // CUDA selector
    cudaSelector.querySelectorAll('.selector-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            cudaSelector.querySelectorAll('.selector-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentSelection.cuda = btn.dataset.value;
            updateAvailableOptions();
        });
    });
    
    // Architecture selector
    archSelector.querySelectorAll('.selector-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            archSelector.querySelectorAll('.selector-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentSelection.arch = btn.dataset.value;
            updateDownloadLink();
        });
    });
}

function updatePythonVersions(versionBtn) {
    const supportedVersions = versionBtn.dataset.pythonVersions.split(',');
    
    // Sort Python versions numerically (e.g., 3.8, 3.9, 3.10, 3.11, 3.12)
    supportedVersions.sort((a, b) => {
        const [aMajor, aMinor] = a.split('.').map(Number);
        const [bMajor, bMinor] = b.split('.').map(Number);
        if (aMajor !== bMajor) return aMajor - bMajor;
        return aMinor - bMinor;
    });
    
    // Clear and rebuild Python selector
    pythonSelector.innerHTML = '';
    
    supportedVersions.forEach((ver, index) => {
        const btn = document.createElement('button');
        btn.className = 'selector-btn';
        btn.dataset.value = ver;
        btn.textContent = ver;
        
        if (index === 0 || ver === currentSelection.python) {
            btn.classList.add('active');
            currentSelection.python = ver;
        }
        
        btn.addEventListener('click', () => {
            pythonSelector.querySelectorAll('.selector-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentSelection.python = btn.dataset.value;
            updateDownloadLink();
        });
        
        pythonSelector.appendChild(btn);
    });
    
    // If current selection is not available, select first
    if (!supportedVersions.includes(currentSelection.python)) {
        const firstBtn = pythonSelector.querySelector('.selector-btn');
        if (firstBtn) {
            firstBtn.classList.add('active');
            currentSelection.python = firstBtn.dataset.value;
        }
    }
}

function updateUbuntuVersions(versionBtn) {
    const supportedVersions = versionBtn.dataset.ubuntuVersions.split(',').filter(v => v);
    
    linuxVersionSelector.querySelectorAll('.selector-btn').forEach(btn => {
        const ubuntuVer = btn.dataset.value;
        if (supportedVersions.includes(ubuntuVer)) {
            btn.style.display = 'inline-block';
            btn.disabled = false;
        } else {
            btn.style.display = 'none';
            btn.disabled = true;
        }
    });
    
    // If current selection is not available, select the first available one
    if (!supportedVersions.includes(currentSelection.linuxVersion)) {
        const firstAvailable = Array.from(linuxVersionSelector.querySelectorAll('.selector-btn')).find(btn => 
            supportedVersions.includes(btn.dataset.value)
        );
        if (firstAvailable) {
            linuxVersionSelector.querySelectorAll('.selector-btn').forEach(b => b.classList.remove('active'));
            firstAvailable.classList.add('active');
            currentSelection.linuxVersion = firstAvailable.dataset.value;
        }
    }
}

function updateAvailableOptions() {
    if (!downloadsData || !currentSelection.version) return;
    
    const versionData = downloadsData[currentSelection.version];
    if (!versionData) return;
    
    // Handle macOS - hide CUDA options
    if (currentSelection.os === 'macos') {
        cudaRow.style.display = 'none';
        currentSelection.cuda = 'cpu';
    } else {
        cudaRow.style.display = 'flex';
    }
    
    // Get available options for current configuration
    let availableData = versionData[currentSelection.os];
    
    // For Linux, both app and wheel are organized by Ubuntu version
    // (wheel packages use different manylinux versions for different Ubuntu versions)
    if (currentSelection.os === 'linux' && currentSelection.linuxVersion) {
        availableData = versionData.linux?.[currentSelection.linuxVersion];
    }
    
    if (!availableData) {
        updateDownloadLink();
        return;
    }
    
    // Update available CUDA options
    const packageData = availableData[currentSelection.package];
    if (packageData && currentSelection.os !== 'macos') {
        const availableCuda = Object.keys(packageData);
        cudaSelector.querySelectorAll('.selector-btn').forEach(btn => {
            const cudaValue = btn.dataset.value;
            if (availableCuda.includes(cudaValue)) {
                btn.style.display = 'inline-block';
                btn.disabled = false;
            } else {
                btn.style.display = 'none';
                btn.disabled = true;
            }
        });
        
        // If current CUDA selection is not available, select first available
        if (!availableCuda.includes(currentSelection.cuda)) {
            const firstAvailable = cudaSelector.querySelector('.selector-btn:not([disabled])');
            if (firstAvailable) {
                cudaSelector.querySelectorAll('.selector-btn').forEach(b => b.classList.remove('active'));
                firstAvailable.classList.add('active');
                currentSelection.cuda = firstAvailable.dataset.value;
            }
        }
    }
    
    // Update available Architecture options
    const cudaData = packageData?.[currentSelection.cuda];
    if (cudaData) {
        const availableArch = Object.keys(cudaData);
        archSelector.querySelectorAll('.selector-btn').forEach(btn => {
            const archValue = btn.dataset.value;
            if (availableArch.includes(archValue)) {
                btn.style.display = 'inline-block';
                btn.disabled = false;
            } else {
                btn.style.display = 'none';
                btn.disabled = true;
            }
        });
        
        // If current arch selection is not available, select first available
        if (!availableArch.includes(currentSelection.arch)) {
            const firstAvailable = archSelector.querySelector('.selector-btn:not([disabled])');
            if (firstAvailable) {
                archSelector.querySelectorAll('.selector-btn').forEach(b => b.classList.remove('active'));
                firstAvailable.classList.add('active');
                currentSelection.arch = firstAvailable.dataset.value;
            }
        }
    }
    
    updateDownloadLink();
}

function updateDownloadLink() {
    if (!downloadsData) return;
    
    const { version, os, package: pkg, linuxVersion, python, cuda, arch } = currentSelection;
    
    let download = null;
    
    try {
        if (pkg === 'app') {
            if (os === 'linux') {
                download = downloadsData[version]?.linux?.[linuxVersion]?.app?.[cuda]?.[arch];
            } else {
                download = downloadsData[version]?.[os]?.app?.[cuda]?.[arch];
            }
        } else if (pkg === 'wheel') {
            if (os === 'linux') {
                // Wheels are organized by Ubuntu version (manylinux versions)
                download = downloadsData[version]?.linux?.[linuxVersion]?.wheel?.[cuda]?.[arch]?.[python];
            } else {
                download = downloadsData[version]?.[os]?.wheel?.[cuda]?.[arch]?.[python];
            }
        }
    } catch (e) {
        console.error('Error finding download:', e);
    }
    
    if (download) {
        const downloadBtn = translations[currentLang]?.['footer.download_btn'] || 'Download';
        downloadOutput.innerHTML = `
            <a href="${download.url}" class="download-link">
                <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                    <path d="M10 15L3 8h4V2h6v6h4l-7 7z"/>
                    <rect y="17" width="20" height="2"/>
                </svg>
                ${downloadBtn}
            </a>
            <div class="file-info">${translations[currentLang]?.['download.file_size'] || 'Size'}: ${download.size}</div>
        `;
    } else {
        const notAvailableMsg = translations[currentLang]?.['download.not_available'] || 'Not available. Please try a different combination of options.';
        downloadOutput.innerHTML = `
            <div class="not-available">
                ${notAvailableMsg}
            </div>
        `;
    }
}

// Initialize download selector when DOM is ready
function initializeDownloadSelector() {
    // Initialize DOM elements
    versionSelector = document.getElementById('version-selector');
    osSelector = document.getElementById('os-selector');
    linuxVersionRow = document.getElementById('linux-version-row');
    linuxVersionSelector = document.getElementById('linux-version-selector');
    packageSelector = document.getElementById('package-selector');
    pythonRow = document.getElementById('python-row');
    pythonSelector = document.getElementById('python-selector');
    cudaRow = document.getElementById('cuda-row');
    cudaSelector = document.getElementById('cuda-selector');
    archRow = document.getElementById('arch-row');
    archSelector = document.getElementById('arch-selector');
    downloadOutput = document.getElementById('download-output');
    
    if (versionSelector) {
        console.log('🔧 Initializing download selector...');
        loadDownloadsData();
    } else {
        console.log('ℹ️ Download selector not found on this page');
    }
}

// Initialize when DOM is fully loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeDownloadSelector);
} else {
    // DOM already loaded
    initializeDownloadSelector();
}

