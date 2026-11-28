# WebGL Dynamic Backgrounds

6 professional WebGL shader-based backgrounds for logo showcases. Each background is fully interactive with mouse tracking and real-time animation.

## Background Styles

### 1. LED Matrix
**Visual**: High-density 90x90 LED grid with flowing wave animation
**Color**: Adaptive theme (Cyber Blue, Amber, Rust Gold, Crimson, Emerald)
**Use Case**: Tech products, digital services, AI/ML platforms
**Technical**: Fragment shader with grid pattern and sine wave animation

### 2. Fluid Warping
**Visual**: Domain warping with Fractional Brownian Motion (FBM)
**Color**: Adaptive theme with 3-color gradient blending
**Use Case**: Creative tools, design software, fluid/organic brands
**Technical**: Multi-layer FBM with mouse interaction and shimmer effects

### 3. Fabric Wave
**Visual**: Silk-like undulation with cross-wave patterns
**Color**: Dark gray base (0.08, 0.09, 0.1) with subtle highlights
**Use Case**: Fashion, luxury, textile, premium products
**Technical**: Sine/cosine wave combination with noise sparkles

### 4. Off-Center Ripple (Corner Ripple)
**Visual**: Dual ripples emanating from opposite corners
**Color**: Medium gray base (0.16, 0.18, 0.23)
**Use Case**: Asymmetric brands, dynamic products, modern tech
**Technical**: Two exponential decay ripples with different frequencies

### 5. Holographic Dispersion
**Visual**: Iridescent fluid with RGB chromatic aberration
**Color**: Dark titanium base with subtle color shifts
**Use Case**: Premium tech, optical products, holographic brands
**Technical**: Palette-based color mapping with dispersion effect
**Key Feature**: RGB channel separation creates prism-like color dispersion

### 6. Spiral Vortex
**Visual**: Rotating spiral with angular momentum
**Color**: Light gray base (0.91, 0.92, 0.93) with color bands
**Use Case**: Motion-focused brands, rotation/cycle concepts, energy
**Technical**: Polar coordinate transformation with spiral distortion

## Technical Implementation

### Shader Structure
All backgrounds use WebGL fragment shaders with:
- `u_resolution`: Canvas dimensions
- `u_time`: Animation time
- `u_mouse`: Mouse position (0-1 normalized)
- Custom uniforms for colors (LED Matrix, Fluid Warping)

### Adaptive Color System
LED Matrix and Fluid Warping use seed-based theme selection:
```javascript
function getAdaptiveTheme(seed) {
    const themes = [
        { dark: [0.02, 0.03, 0.05], bright: [0.15, 0.4, 0.8] },  // Cyber Blue
        { dark: [0.04, 0.015, 0.0], bright: [0.8, 0.3, 0.0] },   // Amber
        { dark: [0.1, 0.03, 0.0], bright: [0.85, 0.35, 0.1] },   // Rust Gold
        { dark: [0.02, 0.01, 0.03], bright: [0.7, 0.05, 0.05] }, // Crimson
        { dark: [0.0, 0.05, 0.03], bright: [0.1, 0.6, 0.4] },    // Emerald
    ];
    return themes[seed % themes.length];
}
```

### Mouse Interaction
All backgrounds respond to mouse movement:
- Ripple effects emanate from cursor
- Fluid distortion follows mouse position
- Exponential decay for natural feel

### Performance
- 60 FPS target on modern hardware
- Efficient shader compilation
- Automatic canvas resize with device pixel ratio
- RequestAnimationFrame for smooth animation

## Usage in Logo Showcase

### When to Use Each Style

**LED Matrix**: Digital products, SaaS platforms, AI tools, blockchain
**Fluid Warping**: Creative software, design tools, artistic brands
**Fabric Wave**: Fashion, textiles, luxury goods, premium services
**Off-Center Ripple**: Modern tech, asymmetric designs, dynamic brands
**Holographic Dispersion**: Optical tech, premium electronics, holographic products
**Spiral Vortex**: Motion/energy brands, circular concepts, rotation themes

### Logo Color Recommendations

**Dark backgrounds** (LED Matrix, Fluid Warping, Fabric Wave, Off-Center, Dispersion):
- Use white or very light logos (#FFFFFF, #F5F5F5)
- Ensure high contrast for readability

**Light backgrounds** (Spiral Vortex):
- Use black or very dark logos (#000000, #1A1A1A)
- Maintain strong contrast

### Integration with Showcase Webpage

The background library provides:
1. **Thumbnail view**: 6 cards in grid layout (16:9 aspect ratio)
2. **Fullscreen mode**: Click any card to expand
3. **Interactive preview**: Mouse tracking in both modes
4. **Logo overlay**: Centered logo on each background
5. **Metadata display**: Style name and technical details

## File Structure

```
logo-generator/
├── references/
│   └── webgl_backgrounds.md          # This file
├── assets/
│   └── background_library.html       # Interactive showcase
└── scripts/
    └── generate_showcase_webpage.py  # Webpage generator
```

## Code Reference

Full shader implementations available in:
- `/Users/guohao/Documents/code/new-skill/background_library_v2.html`

Each shader is ~30-60 lines of GLSL code with detailed comments.
