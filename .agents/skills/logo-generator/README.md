# Logo Generator / Logo 生成器

![Showcase](https://github.com/user-attachments/assets/c36c0e15-370c-4670-906e-36b829f9e7fa)

[English](#english) | [中文](#中文)

---

## English

Professional SVG logo generator with high-end showcase presentations. Generate 6+ design variants based on product characteristics, then create stunning showcase images with 12 professional background styles.

### About

This skill was created to solve a common problem: generating professional logos quickly without sacrificing quality. Traditional logo design requires extensive back-and-forth with designers, while AI-generated logos often lack the refinement and presentation quality needed for real products.

**Logo Generator** bridges this gap by:
- Applying proven design principles (extreme simplicity, generous negative space, precise proportions)
- Generating multiple variants to explore different directions
- Creating production-ready showcase images with professional backgrounds
- Providing both SVG (editable) and PNG (ready-to-use) formats

Perfect for:
- **Startups** needing a professional logo quickly
- **Developers** building side projects
- **Designers** exploring initial concepts
- **Product teams** iterating on brand identity

The skill leverages Gemini 3.1 Flash Image Preview (Nano Banana) to generate high-end showcase images that look like they came from a professional design studio.

### Features

- **SVG Logo Generation**: Create geometric logos with dot matrix, line systems, and mixed compositions
- **Design Variety**: Generate 6+ distinct variants per request with different pattern types
- **Professional Showcase**: 12 curated background styles (void, frosted, fluid, spotlight, analog liquid, LED matrix, editorial, iridescent, morning, clinical, UI container, Swiss flat)
- **Nano Banana Integration**: High-end showcase images using Gemini 3.1 Flash Image Preview
- **Interactive Previews**: Beautiful HTML showcases with hover effects and smooth transitions

### Installation

#### Method 1: Automatic Installation (Recommended)

```bash
npx skills add https://github.com/op7418/logo-generator-skill.git
```

This will automatically install the skill to the correct directory.

#### Method 2: Git Clone

```bash
git clone https://github.com/op7418/logo-generator-skill.git ~/.claude/skills/logo-generator
```

#### Method 3: Manual Installation

1. Download this repository
2. Copy the `logo-generator` folder to your Claude Code skills directory:
   - **macOS/Linux**: `~/.claude/skills/`
   - **Windows**: `%USERPROFILE%\.claude\skills\`
3. Ensure the folder structure contains `SKILL.md` and `README.md`

#### Post-Installation Setup

After installation, set up the required dependencies:

```bash
cd ~/.claude/skills/logo-generator
pip install -r requirements.txt
```

Configure your Gemini API key:

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

Restart Claude Code and verify by typing `/logo-generator` in the conversation.

### Usage

#### Basic Workflow

1. **Start a logo project**:
   ```
   Generate a logo for my AI product called "DataFlow"
   ```

2. **Provide context** (the AI will ask):
   - Industry/Category (e.g., AI, fintech, design tools)
   - Core Concept (e.g., connection, flow, security)
   - Design Preferences (minimal/complex, cold/warm)

3. **Review variants**: The AI generates 6+ SVG logo variants with design rationale

4. **Select and refine**: Choose your favorite, request adjustments

5. **Generate showcase**: Create professional presentation images with multiple background styles

#### Example Commands

```
Create a logo for a blockchain security platform

Generate 6 logo variants for "CloudSync" - a file sync tool

Show me the logo in different background styles

Export the logo as PNG at 2048x2048
```

### Workflow Phases

**Phase 1: Information Gathering**  
Collect product name, industry, core concept, and design preferences

**Phase 2: Pattern Matching & SVG Generation**  
- Generate 6+ distinct design variants
- Create interactive HTML showcase
- Explain design rationale for each variant

**Phase 3: Iteration & Refinement**  
- Select favorite variants
- Adjust parameters (size, spacing, rotation)
- Combine elements from different variants

**Phase 4: High-End Showcase Generation**  
- Export SVG to PNG (1024x1024px)
- Select 4 showcase styles based on product type
- Generate showcase images with Nano Banana
- Create final presentation webpage

**Phase 5: Delivery**  
- Interactive HTML showcase page
- SVG files (editable vector format)
- PNG exports (various sizes)
- Showcase images (4 professional backgrounds)

### Background Styles

#### Dark Styles (6)
- **The Void** - Absolute black with silver micro noise (hardcore tech)
- **Frosted Horizon** - Titanium gray with organic texture (premium products)
- **Fluid Abyss** - Deep purple with fluid fusion (AI-native)
- **Studio Spotlight** - Carbon gray with editorial lighting (magazine quality)
- **Analog Liquid** - Metallic shimmer on solid color base (creative brands)
- **LED Matrix** - Digital retro with glowing dots (cyberpunk)

#### Light Styles (6)
- **Editorial Paper** - Off-white with paper texture (humanistic brands)
- **Iridescent Frost** - Silver-gray with holographic hints (tech hardware)
- **Morning Aura** - Warm ivory with pastel colors (approachable AI)
- **Clinical Studio** - Pure white with geometric shadows (algorithm-driven)
- **UI Container** - Frosted glass container effect (SaaS platforms)
- **Swiss Flat** - Pure solid color, zero effects (timeless authority)

### Design Principles

1. **Extreme Simplicity** - 1-2 core elements maximum
2. **Generous Negative Space** - At least 40-50% empty canvas
3. **Precise Proportions** - Line weights 2.5-4px, proper spacing
4. **Visual Tension** - Intentional asymmetry creates interest
5. **Restraint Over Decoration** - Every element must justify its existence
6. **Single Focal Point** - Clear visual hierarchy

### File Structure

```
logo-generator/
├── SKILL.md                    # Skill definition and workflow
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── scripts/
│   ├── svg_to_png.py          # SVG to PNG converter
│   └── generate_showcase.py   # Showcase image generator
├── references/
│   ├── design_patterns.md     # Comprehensive design guide
│   ├── background_styles.md   # Background style specifications
│   └── webgl_backgrounds.md   # WebGL dynamic backgrounds
└── assets/
    ├── showcase_template.html # HTML template for showcases
    └── background_library.html # Interactive WebGL backgrounds
```

### Requirements

- Python 3.8+
- Dependencies: `google-genai`, `python-dotenv`, `cairosvg`, `Pillow`
- Gemini API key (for showcase generation)

### API Configuration

#### Official Google Gemini API

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.1-flash-image-preview
```

#### Third-Party API Endpoint

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_API_BASE_URL=https://api.example.com/v1
GEMINI_MODEL=gemini-3.1-flash-image-preview
```

### License

MIT License - feel free to use for personal or commercial projects

### Credits

- Design patterns inspired by modern brand identity systems
- Showcase styles curated from high-end design presentations
- Powered by Gemini 3.1 Flash Image Preview (Nano Banana)

### Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

---

## 中文

专业的 SVG Logo 生成器，配备高端展示效果。基于产品特性生成 6+ 个设计变体，并使用 12 种专业背景风格创建精美的展示图。

### 关于本项目

这个技能的诞生源于一个常见痛点：如何在不牺牲质量的前提下快速生成专业 Logo。传统设计流程需要与设计师反复沟通，而 AI 生成的 Logo 往往缺乏精致度和展示效果。

**Logo Generator** 通过以下方式解决这个问题：
- 应用经过验证的设计原则（极致简洁、慷慨留白、精准比例）
- 生成多个变体以探索不同方向
- 创建可直接使用的专业展示图
- 提供 SVG（可编辑）和 PNG（即用）两种格式

适用场景：
- **创业公司** 需要快速获得专业 Logo
- **开发者** 构建个人项目
- **设计师** 探索初期概念
- **产品团队** 迭代品牌形象

本技能利用 Gemini 3.1 Flash Image Preview (Nano Banana) 生成高端展示图，效果媲美专业设计工作室。

### 功能特性

- **SVG Logo 生成**：创建几何 Logo，支持点阵、线条系统和混合构图
- **设计多样性**：每次请求生成 6+ 个不同范式的设计变体
- **专业展示**：12 种精选背景风格（绝对虚空、磨砂穹顶、流体深渊、聚光灯、物理流体、LED 矩阵、编辑纸张、幻彩透砂、晨曦光晕、临床工作室、UI 容器、瑞士扁平）
- **Nano Banana 集成**：使用 Gemini 3.1 Flash Image Preview 生成高端展示图
- **交互式预览**：精美的 HTML 展示页面，带悬停效果和流畅过渡

### 安装方式

#### 方法 1：自动安装（推荐）

```bash
npx skills add https://github.com/op7418/logo-generator-skill.git
```

这将自动将技能安装到正确的目录。

#### 方法 2：Git 克隆

```bash
git clone https://github.com/op7418/logo-generator-skill.git ~/.claude/skills/logo-generator
```

#### 方法 3：手动安装

1. 下载本仓库
2. 将 `logo-generator` 文件夹复制到 Claude Code 技能目录：
   - **macOS/Linux**：`~/.claude/skills/`
   - **Windows**：`%USERPROFILE%\.claude\skills\`
3. 确保文件夹结构包含 `SKILL.md` 和 `README.md`

#### 安装后配置

安装完成后，设置所需的依赖：

```bash
cd ~/.claude/skills/logo-generator
pip install -r requirements.txt
```

配置 Gemini API 密钥：

```bash
cp .env.example .env
# 编辑 .env 文件并添加你的 GEMINI_API_KEY
```

重启 Claude Code 并在对话中输入 `/logo-generator` 验证安装成功。

### 使用方法

#### 基本工作流

1. **启动 Logo 项目**：
   ```
   为我的 AI 产品 "DataFlow" 生成一个 Logo
   ```

2. **提供上下文**（AI 会询问）：
   - 行业/类别（如：AI、金融科技、设计工具）
   - 核心概念（如：连接、流动、安全）
   - 设计偏好（极简/复杂、冷色/暖色）

3. **查看变体**：AI 生成 6+ 个 SVG Logo 变体并说明设计理念

4. **选择和优化**：选择你喜欢的方案，请求调整

5. **生成展示**：创建多种背景风格的专业展示图

#### 示例命令

```
为区块链安全平台创建一个 Logo

为 "CloudSync" 文件同步工具生成 6 个 Logo 变体

展示不同背景风格下的 Logo 效果

导出 2048x2048 的 PNG 格式 Logo
```

### 工作流程阶段

**阶段 1：信息收集**  
收集产品名称、行业、核心概念和设计偏好

**阶段 2：范式匹配与 SVG 生成**  
- 生成 6+ 个不同的设计变体
- 创建交互式 HTML 展示页面
- 解释每个变体的设计理念

**阶段 3：迭代与优化**  
- 选择喜欢的变体
- 调整参数（大小、间距、旋转）
- 组合不同变体的元素

**阶段 4：高端展示图生成**  
- 导出 SVG 为 PNG（1024x1024px）
- 根据产品类型选择 4 种展示风格
- 使用 Nano Banana 生成展示图
- 创建最终展示网页

**阶段 5：交付**  
- 交互式 HTML 展示页面
- SVG 文件（可编辑矢量格式）
- PNG 导出（多种尺寸）
- 展示图（4 种专业背景）

### 背景风格

#### 深色风格（6 种）
- **绝对虚空** - 纯黑 + 银色微噪点（硬核科技）
- **磨砂穹顶** - 钛灰 + 有机纹理（高端产品）
- **流体深渊** - 深紫 + 流体融合（AI 原生）
- **聚光灯** - 碳灰 + 编辑光效（杂志质感）
- **物理流体** - 纯色底 + 金属质感（创意品牌）
- **LED 矩阵** - 数字复古 + 发光点阵（赛博朋克）

#### 浅色风格（6 种）
- **编辑纸张** - 米白 + 纸张纹理（人文品牌）
- **幻彩透砂** - 银灰 + 全息微光（科技硬件）
- **晨曦光晕** - 暖象牙 + 柔和色彩（亲和 AI）
- **临床工作室** - 纯白 + 几何阴影（算法驱动）
- **UI 容器** - 磨砂玻璃容器效果（SaaS 平台）
- **瑞士扁平** - 纯色无效果（永恒权威）

### 设计原则

1. **极致简洁** - 最多 1-2 个核心元素
2. **慷慨留白** - 至少 40-50% 空白画布
3. **精准比例** - 线条粗细 2.5-4px，合理间距
4. **视觉张力** - 有意的不对称创造趣味
5. **克制而非装饰** - 每个元素必须证明其存在价值
6. **单一焦点** - 清晰的视觉层级

### 文件结构

```
logo-generator/
├── SKILL.md                    # Skill 定义和工作流程
├── README.md                   # 本文件
├── requirements.txt            # Python 依赖
├── .env.example               # 环境变量模板
├── scripts/
│   ├── svg_to_png.py          # SVG 转 PNG 工具
│   └── generate_showcase.py   # 展示图生成器
├── references/
│   ├── design_patterns.md     # 综合设计指南
│   ├── background_styles.md   # 背景风格规范
│   └── webgl_backgrounds.md   # WebGL 动态背景
└── assets/
    ├── showcase_template.html # 展示页面模板
    └── background_library.html # 交互式 WebGL 背景库
```

### 系统要求

- Python 3.8+
- 依赖：`google-genai`、`python-dotenv`、`cairosvg`、`Pillow`
- Gemini API 密钥（用于展示图生成）

### API 配置

#### 官方 Google Gemini API

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.1-flash-image-preview
```

#### 第三方 API 端点

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_API_BASE_URL=https://api.example.com/v1
GEMINI_MODEL=gemini-3.1-flash-image-preview
```

### 开源协议

MIT License - 可自由用于个人或商业项目

### 致谢

- 设计范式灵感来自现代品牌识别系统
- 展示风格精选自高端设计展示
- 由 Gemini 3.1 Flash Image Preview (Nano Banana) 驱动

### 贡献

欢迎贡献！请随时提交 issue 或 pull request。
