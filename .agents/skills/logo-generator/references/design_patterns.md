# Logo Design Patterns — Comprehensive Guide

This document is the core design reference for SVG logo generation. It provides detailed design philosophies, concrete SVG implementations, and combination strategies. When generating logos, use this document to ensure variety, quality, and meaningful design.

---

## Part 0: Critical Design Principles (READ THIS FIRST)

### What Makes a Logo "High-End"?

After analyzing 100+ reference logos, these are the non-negotiable principles:

#### 1. Extreme Simplicity
- **Rule**: Each logo should have 1-2 core elements maximum
- **Bad**: 15 hexagons in a grid (too busy, no focal point)
- **Good**: A single thick circle (Void Circle), or 2 merging nodes (Gooey Nodes)
- **Why**: Simplicity creates memorability and scalability

#### 2. Generous Negative Space
- **Rule**: At least 40-50% of the canvas should be empty
- **Bad**: Elements packed tightly, no breathing room
- **Good**: Large void at center, elements positioned with intentional gaps
- **Why**: Negative space is not "wasted space"—it's part of the design

#### 3. Precise Proportions
- **Rule**: Line weights should be 2.5-4px (in viewBox 0 0 100 100)
- **Rule**: Dot sizes should be 2-8px radius
- **Rule**: Spacing between elements should be at least 8-12px
- **Bad**: strokeWidth="1.5" looks weak, strokeWidth="6" looks clumsy
- **Good**: strokeWidth="3" or "4" for primary elements, "2" for secondary
- **Why**: Precision creates professionalism

#### 4. Visual Tension Through Asymmetry
- **Rule**: Perfect symmetry is boring—introduce intentional imbalance
- **Bad**: Perfectly centered, perfectly symmetric grid
- **Good**: Asymmetric dot cluster (heavy bottom-left, light top-right)
- **Why**: Asymmetry creates visual interest and dynamic energy

#### 5. Restraint Over Decoration
- **Rule**: Every element must justify its existence
- **Bad**: Adding gradients, shadows, multiple colors "because we can"
- **Good**: Pure black on white/light gray, no effects except subtle hover animations
- **Why**: Restraint signals confidence and sophistication

#### 6. Single Focal Point
- **Rule**: The eye should know where to look immediately
- **Bad**: Multiple competing elements of equal visual weight
- **Good**: One dominant element (large circle, main node cluster, central burst)
- **Why**: Clear hierarchy guides perception

#### 7. Structural Stability
- **Rule**: Logos must have visual weight and structural integrity
- **Bad**: 2-3 thin lines (strokeWidth < 2.5) with no solid mass—feels fragile and weak
- **Good**: Dense line systems (6-8 lines), solid shapes, or thick strokes (strokeWidth 3-4)
- **Why**: Structural stability ensures the logo feels grounded and professional
- **Exception**: If using line systems, use dense repetition (6+ lines) to create visual mass

#### 8. Rounded Negative Space Cuts
- **Rule**: When using negative space (cutouts/subtractions), openings must be rounded
- **Bad**: Sharp corners or pointed edges on cutout areas—feels harsh and unfinished
- **Good**: Circular or rounded-rectangle cutouts with smooth edges
- **Why**: Rounded cuts feel intentional and refined; sharp cuts feel accidental

### Reference Analysis Summary

From analyzing the provided reference images:

**09b239bb127ab20616631d72d7fe0e82.jpg** (Infinity Loop)
- Single smooth closed path, strokeWidth ~4-5
- Extreme simplicity, pure form
- Lesson: One perfect element beats five mediocre ones

**b541c30ccc1e22b83984a22890337437.jpg** (Gooey Nodes)
- 2 main nodes + 2-3 small satellites
- Asymmetric placement, organic feel
- Lesson: Use SVG filters (goo effect) for sophistication

**282b7bd3919f1b1646e8d8bae149d347.jpg** (Radial Burst)
- 8 rays from center, alternating lengths
- Precise geometry, strong focal point
- Lesson: Rhythm through repetition with variation

**4221df1d4cbb25fce1d4a54e35cb9d3e.jpg** (Punched Circle)
- Solid circle + asymmetric perforation grid
- Negative space as design element
- Lesson: Subtract rather than add

**009244dc6bbce8a123f37e332e36be72.jpg** (Star Burst)
- Radial lines with gaps forming negative star
- Negative space creates the symbol
- Lesson: What you don't draw is as important as what you do

### Design Checklist (Use Before Finalizing)

Before presenting a logo, verify:

- [ ] **Element count**: 1-2 core elements (max 5-6 total shapes)
- [ ] **Negative space**: At least 40% of canvas is empty
- [ ] **Line weight**: Primary elements use strokeWidth 2.5-4
- [ ] **Focal point**: Clear visual hierarchy, eye knows where to look
- [ ] **Asymmetry**: Intentional imbalance creates interest
- [ ] **Scalability**: Works at 16x16 (favicon) and 512x512
- [ ] **Restraint**: No unnecessary decoration or effects
- [ ] **Breathing room**: Minimum 8-12px spacing between elements

---

## Part 1: Fundamental Design Elements

### 1.1 Dot Matrix System

Dot matrix is one of the most versatile and SVG-friendly patterns. The key is choosing the right dot shape and arrangement to convey meaning.

#### Circle Dots — Concentric Ring Layout

**Philosophy**: Concentric circles radiate outward from a center point, suggesting origin, source, or emanation. The center void creates tension between presence and absence.

**Key parameters**:
- Ring count: 2-4 rings
- Dots per ring: follows ratio (inner fewer, outer more), e.g., 6 / 12 / 18
- Dot radius: decreasing outward (e.g., 3 → 2.5 → 2) creates depth
- Center: intentionally left empty (negative space)

```svg
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <g fill="currentColor">
    <!-- Ring 1: 6 dots, radius 15 from center, dot size 3 -->
    <circle cx="50" cy="35" r="3"/>
    <circle cx="62.99" cy="42.5" r="3"/>
    <circle cx="62.99" cy="57.5" r="3"/>
    <circle cx="50" cy="65" r="3"/>
    <circle cx="37.01" cy="57.5" r="3"/>
    <circle cx="37.01" cy="42.5" r="3"/>
    
    <!-- Ring 2: 12 dots, radius 25 from center, dot size 2.5 -->
    <circle cx="50" cy="25" r="2.5"/>
    <circle cx="62.5" cy="28.35" r="2.5"/>
    <circle cx="71.65" cy="37.5" r="2.5"/>
    <circle cx="75" cy="50" r="2.5"/>
    <circle cx="71.65" cy="62.5" r="2.5"/>
    <circle cx="62.5" cy="71.65" r="2.5"/>
    <circle cx="50" cy="75" r="2.5"/>
    <circle cx="37.5" cy="71.65" r="2.5"/>
    <circle cx="28.35" cy="62.5" r="2.5"/>
    <circle cx="25" cy="50" r="2.5"/>
    <circle cx="28.35" cy="37.5" r="2.5"/>
    <circle cx="37.5" cy="28.35" r="2.5"/>
  </g>
</svg>
```

**Suitable for**: AI, data platforms, origin/source concepts, scientific tools

#### Rounded Rectangle Dots — Grid Layout

**Philosophy**: Rounded rectangles convey modularity and structure. Their softened corners make them friendlier than sharp squares while retaining a sense of order and precision.

**Key parameters**:
- Grid: typically 3x3, 4x4, or 5x5
- Corner radius: 30-50% of shorter side (too round → circles, too sharp → rigid)
- Spacing: consistent gaps between elements
- Selective removal: remove certain dots to create a recognizable shape or letter

```svg
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <g fill="currentColor">
    <!-- 4x4 grid of rounded rectangles, some removed to form "L" shape -->
    <!-- Row 1 -->
    <rect x="20" y="15" width="10" height="10" rx="3"/>
    <rect x="35" y="15" width="10" height="10" rx="3"/>
    <!-- Row 2 -->
    <rect x="20" y="30" width="10" height="10" rx="3"/>
    <rect x="35" y="30" width="10" height="10" rx="3"/>
    <!-- Row 3 -->
    <rect x="20" y="45" width="10" height="10" rx="3"/>
    <rect x="35" y="45" width="10" height="10" rx="3"/>
    <!-- Row 4 - full row -->
    <rect x="20" y="60" width="10" height="10" rx="3"/>
    <rect x="35" y="60" width="10" height="10" rx="3"/>
    <rect x="50" y="60" width="10" height="10" rx="3"/>
    <rect x="65" y="60" width="10" height="10" rx="3"/>
  </g>
</svg>
```

**Suitable for**: Tech platforms, developer tools, modular systems, dashboard products

#### Capsule Dots — Path-Aligned Layout

**Philosophy**: Capsules (elongated rounded rectangles) suggest motion and directionality. When arranged along a curve, they create a sense of flow and progress.

**Key parameters**:
- Aspect ratio: typically 2:1 to 3:1 (width:height)
- Rotation: each capsule rotates to follow the path tangent
- Size variation: larger near center, smaller at edges
- Spacing: slightly irregular for organic feel

```svg
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <g fill="currentColor">
    <!-- Capsules flowing along an arc path -->
    <rect x="22" y="62" width="14" height="6" rx="3" transform="rotate(-45, 29, 65)"/>
    <rect x="32" y="50" width="16" height="7" rx="3.5" transform="rotate(-30, 40, 53.5)"/>
    <rect x="43" y="42" width="18" height="8" rx="4" transform="rotate(-10, 52, 46)"/>
    <rect x="57" y="40" width="16" height="7" rx="3.5" transform="rotate(10, 65, 43.5)"/>
    <rect x="68" y="44" width="14" height="6" rx="3" transform="rotate(30, 75, 47)"/>
    <rect x="76" y="52" width="12" height="5" rx="2.5" transform="rotate(50, 82, 54.5)"/>
  </g>
</svg>
```

**Suitable for**: Workflow tools, collaboration platforms, streaming/data pipeline products

#### Hexagon Dots — Honeycomb Layout

**Philosophy**: Hexagons tessellate perfectly, suggesting efficient organization and distributed networks. Natural and mathematical at the same time.

```svg
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <polygon id="hex" points="5,0 2.5,4.33 -2.5,4.33 -5,0 -2.5,-4.33 2.5,-4.33" fill="currentColor"/>
  </defs>
  <g>
    <!-- Row 1 (offset) -->
    <use href="#hex" x="38" y="30"/>
    <use href="#hex" x="50" y="30"/>
    <use href="#hex" x="62" y="30"/>
    <!-- Row 2 -->
    <use href="#hex" x="32" y="40"/>
    <use href="#hex" x="44" y="40"/>
    <use href="#hex" x="56" y="40"/>
    <use href="#hex" x="68" y="40"/>
    <!-- Row 3 (offset) -->
    <use href="#hex" x="38" y="50"/>
    <use href="#hex" x="50" y="50"/>
    <use href="#hex" x="62" y="50"/>
    <!-- Row 4 -->
    <use href="#hex" x="32" y="60"/>
    <use href="#hex" x="44" y="60"/>
    <use href="#hex" x="56" y="60"/>
    <use href="#hex" x="68" y="60"/>
    <!-- Row 5 (offset) -->
    <use href="#hex" x="38" y="70"/>
    <use href="#hex" x="50" y="70"/>
    <use href="#hex" x="62" y="70"/>
  </g>
</svg>
```

**Suitable for**: Distributed systems, blockchain, network infrastructure, community platforms

---

### 1.2 Geometric Shapes

#### Circle-Based Compositions

**Concentric Rings**: Multiple rings of different stroke widths create depth and layering.

```svg
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <g fill="none" stroke="currentColor">
    <circle cx="50" cy="50" r="35" stroke-width="2"/>
    <circle cx="50" cy="50" r="25" stroke-width="1.5"/>
    <circle cx="50" cy="50" r="15" stroke-width="1"/>
    <!-- Accent: filled center dot -->
    <circle cx="50" cy="50" r="4" fill="currentColor" stroke="none"/>
  </g>
</svg>
```

**Arc Segments**: Broken circles create dynamism and suggest motion/progress.

```svg
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <g fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round">
    <!-- Outer arc: 270 degrees -->
    <path d="M 50 15 A 35 35 0 1 1 15 50"/>
    <!-- Inner arc: 180 degrees, rotated -->
    <path d="M 70 50 A 20 20 0 1 1 30 50"/>
    <!-- Center dot -->
    <circle cx="50" cy="50" r="4" fill="currentColor" stroke="none"/>
  </g>
</svg>
```

**Suitable for**: Iteration, cycles, processing, completeness

#### Boolean Operations — Shape Intersection

**Philosophy**: When two shapes overlap, the intersection creates a new, unexpected form. This represents synthesis, combination, or the emergence of something new from existing elements.

```svg
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <clipPath id="circle-clip">
      <circle cx="58" cy="50" r="22"/>
    </clipPath>
  </defs>
  <g fill="currentColor">
    <!-- Left circle outline -->
    <circle cx="42" cy="50" r="22" fill="none" stroke="currentColor" stroke-width="2"/>
    <!-- Right circle outline -->
    <circle cx="58" cy="50" r="22" fill="none" stroke="currentColor" stroke-width="2"/>
    <!-- Intersection area (filled) -->
    <circle cx="42" cy="50" r="22" clip-path="url(#circle-clip)" fill="currentColor" opacity="0.15"/>
  </g>
</svg>
```

**Suitable for**: Collaboration, merging, integration platforms, API gateways

#### Polygon Compositions

**Nested Polygons**: Shapes within shapes create visual depth.

```svg
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <g fill="none" stroke="currentColor" stroke-width="1.5">
    <!-- Outer hexagon -->
    <polygon points="50,18 76,34 76,66 50,82 24,66 24,34"/>
    <!-- Inner hexagon (rotated 30 degrees) -->
    <polygon points="50,28 67,39 67,61 50,72 33,61 33,39"/>
    <!-- Center triangle -->
    <polygon points="50,38 58,52 42,52" fill="currentColor" stroke="none"/>
  </g>
</svg>
```

**Suitable for**: Security, stability, layered systems, frameworks

---

### 1.3 Line System

#### Horizontal Line Fill — Circle Mask

**Philosophy**: Parallel lines filling a circular boundary represent data flow, code structure, or ordered content within a container. Varying line thickness creates emphasis and hierarchy.

```svg
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <clipPath id="circle-mask">
      <circle cx="50" cy="50" r="32"/>
    </clipPath>
  </defs>
  <!-- Lines clipped to circle -->
  <g clip-path="url(#circle-mask)" stroke="currentColor" stroke-linecap="round">
    <line x1="15" y1="26" x2="85" y2="26" stroke-width="1.5"/>
    <line x1="15" y1="32" x2="85" y2="32" stroke-width="1.5"/>
    <line x1="15" y1="38" x2="85" y2="38" stroke-width="2"/>
    <line x1="15" y1="44" x2="85" y2="44" stroke-width="2"/>
    <line x1="15" y1="50" x2="85" y2="50" stroke-width="2.5"/>
    <line x1="15" y1="56" x2="85" y2="56" stroke-width="2.5"/>
    <line x1="15" y1="62" x2="85" y2="62" stroke-width="2"/>
    <line x1="15" y1="68" x2="85" y2="68" stroke-width="2"/>
    <line x1="15" y1="74" x2="85" y2="74" stroke-width="1.5"/>
  </g>
  <!-- Outer ring (thin, dashed) -->
  <circle cx="50" cy="50" r="36" fill="none" stroke="currentColor" stroke-width="0.8" stroke-dasharray="2 3"/>
</svg>
```

**Suitable for**: Code editors, text/content platforms, data processing tools

#### Wave / Curve System

**Philosophy**: Waves suggest rhythm, frequency, and continuous motion. Multiple overlapping waves create complexity from simplicity.

```svg
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <g fill="none" stroke="currentColor" stroke-linecap="round">
    <!-- Wave 1: large amplitude -->
    <path d="M 15 50 Q 30 30, 50 50 T 85 50" stroke-width="2.5"/>
    <!-- Wave 2: medium amplitude, offset -->
    <path d="M 15 50 Q 30 40, 50 50 T 85 50" stroke-width="1.8" opacity="0.6" transform="translate(0, -8)"/>
    <!-- Wave 3: small amplitude, opposite offset -->
    <path d="M 15 50 Q 30 44, 50 50 T 85 50" stroke-width="1.2" opacity="0.4" transform="translate(0, 8)"/>
    <!-- Accent dots at key points -->
    <circle cx="15" cy="50" r="2.5" fill="currentColor" stroke="none"/>
    <circle cx="85" cy="50" r="2.5" fill="currentColor" stroke="none"/>
  </g>
</svg>
```

**Suitable for**: Audio/music, signal processing, communication, analytics

#### Spiral / Fibonacci

**Philosophy**: Spirals represent growth, evolution, and natural order. A mathematical spiral conveys both beauty and precision.

```svg
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
    <!-- Approximated golden spiral using arcs -->
    <path d="M 50 50 
             A 5 5 0 0 1 55 50 
             A 8 8 0 0 1 55 58 
             A 13 13 0 0 1 42 58 
             A 21 21 0 0 1 42 37 
             A 34 34 0 0 1 76 37"/>
    <!-- Center point -->
    <circle cx="50" cy="50" r="2" fill="currentColor" stroke="none"/>
  </g>
</svg>
```

**Suitable for**: Growth, evolution, design tools, creative platforms

---

### 1.4 Node Network

#### Basic Topology

**Philosophy**: Nodes and edges represent connections, relationships, and systems. The hierarchy of node sizes conveys importance, while edge curvature adds elegance.

```svg
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <g>
    <!-- Edges (curved connections) -->
    <path d="M 30 70 Q 50 70, 50 50" stroke="currentColor" stroke-width="2" fill="none"/>
    <path d="M 50 50 Q 50 30, 70 30" stroke="currentColor" stroke-width="2" fill="none"/>
    <path d="M 30 70 Q 20 40, 35 35" stroke="currentColor" stroke-width="1.5" fill="none" opacity="0.5"/>
    <path d="M 35 35 Q 50 35, 50 50" stroke="currentColor" stroke-width="1.5" fill="none" opacity="0.5"/>
    
    <!-- Primary nodes (large) -->
    <circle cx="30" cy="70" r="5" fill="currentColor"/>
    <circle cx="50" cy="50" r="6" fill="currentColor"/>
    <circle cx="70" cy="30" r="5" fill="currentColor"/>
    
    <!-- Secondary nodes (small, subtle) -->
    <circle cx="35" cy="35" r="2.5" fill="currentColor" opacity="0.4"/>
    <circle cx="65" cy="55" r="2" fill="currentColor" opacity="0.3"/>
  </g>
</svg>
```

**Suitable for**: Agent frameworks, network infrastructure, graph databases, social platforms

---

## Part 2: Advanced Combination Strategies

### 2.1 Dot Matrix + Geometric Shape (Filling)

**Concept**: Use dots to fill a geometric shape. The shape provides structure, the dots provide texture.

**Example: Circle dots filling a rounded triangle**

```svg
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <clipPath id="tri-clip">
      <path d="M 50 20 Q 82 68, 75 72 L 25 72 Q 18 68, 50 20 Z"/>
    </clipPath>
  </defs>
  <g clip-path="url(#tri-clip)" fill="currentColor">
    <!-- Grid of dots, clipped to triangle -->
    <circle cx="50" cy="30" r="2.5"/>
    <circle cx="42" cy="40" r="2.5"/>
    <circle cx="50" cy="40" r="2.5"/>
    <circle cx="58" cy="40" r="2.5"/>
    <circle cx="34" cy="50" r="2.5"/>
    <circle cx="42" cy="50" r="2.5"/>
    <circle cx="50" cy="50" r="2.5"/>
    <circle cx="58" cy="50" r="2.5"/>
    <circle cx="66" cy="50" r="2.5"/>
    <circle cx="28" cy="60" r="2.5"/>
    <circle cx="36" cy="60" r="2.5"/>
    <circle cx="44" cy="60" r="2.5"/>
    <circle cx="52" cy="60" r="2.5"/>
    <circle cx="60" cy="60" r="2.5"/>
    <circle cx="68" cy="60" r="2.5"/>
    <circle cx="30" cy="70" r="2.5"/>
    <circle cx="38" cy="70" r="2.5"/>
    <circle cx="46" cy="70" r="2.5"/>
    <circle cx="54" cy="70" r="2.5"/>
    <circle cx="62" cy="70" r="2.5"/>
    <circle cx="70" cy="70" r="2.5"/>
  </g>
</svg>
```

### 2.2 Dot Matrix + Lines (Connected Dots)

**Concept**: Dots serve as nodes, lines connect them. This combines the discrete nature of dots with the continuity of lines.

**Example: Rounded rectangle dots with connecting lines**

```svg
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <g>
    <!-- Connection lines (behind dots) -->
    <line x1="30" y1="35" x2="50" y2="35" stroke="currentColor" stroke-width="1.5"/>
    <line x1="50" y1="35" x2="70" y2="35" stroke="currentColor" stroke-width="1.5"/>
    <line x1="30" y1="50" x2="50" y2="50" stroke="currentColor" stroke-width="1.5"/>
    <line x1="50" y1="50" x2="70" y2="50" stroke="currentColor" stroke-width="1.5"/>
    <line x1="30" y1="65" x2="50" y2="65" stroke="currentColor" stroke-width="1.5"/>
    <line x1="50" y1="65" x2="70" y2="65" stroke="currentColor" stroke-width="1.5"/>
    <!-- Vertical connections -->
    <line x1="30" y1="35" x2="30" y2="65" stroke="currentColor" stroke-width="1"/>
    <line x1="50" y1="35" x2="50" y2="65" stroke="currentColor" stroke-width="1"/>
    <line x1="70" y1="35" x2="70" y2="65" stroke="currentColor" stroke-width="1"/>
    <!-- Diagonal accent -->
    <path d="M 30 35 Q 50 50, 70 65" stroke="currentColor" stroke-width="2" fill="none" stroke-dasharray="4 2"/>
    
    <!-- Rounded rectangle nodes -->
    <rect x="24" y="29" width="12" height="12" rx="3" fill="currentColor"/>
    <rect x="44" y="29" width="12" height="12" rx="3" fill="currentColor"/>
    <rect x="64" y="29" width="12" height="12" rx="3" fill="currentColor"/>
    <rect x="24" y="44" width="12" height="12" rx="3" fill="currentColor" opacity="0.6"/>
    <rect x="44" y="44" width="12" height="12" rx="3" fill="currentColor"/>
    <rect x="64" y="44" width="12" height="12" rx="3" fill="currentColor" opacity="0.6"/>
    <rect x="24" y="59" width="12" height="12" rx="3" fill="currentColor"/>
    <rect x="44" y="59" width="12" height="12" rx="3" fill="currentColor"/>
    <rect x="64" y="59" width="12" height="12" rx="3" fill="currentColor"/>
  </g>
</svg>
```

### 2.3 Geometry + Lines (Outlined Shapes)

**Concept**: Geometric shapes defined by lines rather than fills. This creates an open, technical feel.

**Example: Stacked rotating squares with line connections**

```svg
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <g fill="none" stroke="currentColor">
    <!-- Square 1: 0 degrees -->
    <rect x="30" y="30" width="40" height="40" stroke-width="2"/>
    <!-- Square 2: 22.5 degrees -->
    <rect x="30" y="30" width="40" height="40" stroke-width="1.5" transform="rotate(22.5, 50, 50)"/>
    <!-- Square 3: 45 degrees -->
    <rect x="30" y="30" width="40" height="40" stroke-width="1" transform="rotate(45, 50, 50)" opacity="0.6"/>
    <!-- Center accent -->
    <circle cx="50" cy="50" r="3" fill="currentColor" stroke="none"/>
    <!-- Corner dots on outer square -->
    <circle cx="30" cy="30" r="2" fill="currentColor" stroke="none"/>
    <circle cx="70" cy="30" r="2" fill="currentColor" stroke="none"/>
    <circle cx="70" cy="70" r="2" fill="currentColor" stroke="none"/>
    <circle cx="30" cy="70" r="2" fill="currentColor" stroke="none"/>
  </g>
</svg>
```

### 2.4 Layered Composition (Background + Foreground)

**Concept**: A subtle background pattern (low opacity) with a bold foreground element. Creates depth and visual richness.

**Example: Dot matrix background with bold geometric foreground**

```svg
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <!-- Background: subtle dot grid -->
  <g fill="currentColor" opacity="0.12">
    <circle cx="20" cy="20" r="1.5"/><circle cx="35" cy="20" r="1.5"/>
    <circle cx="50" cy="20" r="1.5"/><circle cx="65" cy="20" r="1.5"/>
    <circle cx="80" cy="20" r="1.5"/>
    <circle cx="20" cy="35" r="1.5"/><circle cx="35" cy="35" r="1.5"/>
    <circle cx="50" cy="35" r="1.5"/><circle cx="65" cy="35" r="1.5"/>
    <circle cx="80" cy="35" r="1.5"/>
    <circle cx="20" cy="50" r="1.5"/><circle cx="35" cy="50" r="1.5"/>
    <circle cx="65" cy="50" r="1.5"/><circle cx="80" cy="50" r="1.5"/>
    <circle cx="20" cy="65" r="1.5"/><circle cx="35" cy="65" r="1.5"/>
    <circle cx="50" cy="65" r="1.5"/><circle cx="65" cy="65" r="1.5"/>
    <circle cx="80" cy="65" r="1.5"/>
    <circle cx="20" cy="80" r="1.5"/><circle cx="35" cy="80" r="1.5"/>
    <circle cx="50" cy="80" r="1.5"/><circle cx="65" cy="80" r="1.5"/>
    <circle cx="80" cy="80" r="1.5"/>
  </g>
  
  <!-- Foreground: bold circle with inner detail -->
  <g>
    <circle cx="50" cy="50" r="22" fill="none" stroke="currentColor" stroke-width="2.5"/>
    <circle cx="50" cy="50" r="8" fill="currentColor"/>
    <!-- Radiating lines -->
    <line x1="50" y1="28" x2="50" y2="36" stroke="currentColor" stroke-width="2"/>
    <line x1="50" y1="64" x2="50" y2="72" stroke="currentColor" stroke-width="2"/>
    <line x1="28" y1="50" x2="36" y2="50" stroke="currentColor" stroke-width="2"/>
    <line x1="64" y1="50" x2="72" y2="50" stroke="currentColor" stroke-width="2"/>
  </g>
</svg>
```

---

## Part 3: Letter / Symbol Integration

### 3.1 Geometric Letter Abstraction

**Philosophy**: Extract the essential geometry from a letter and reconstruct it using pure geometric elements. Don't try to render a full font — instead, capture the structural DNA of the letter.

**Example: Letter "P" as geometric construction**

```svg
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <g fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round">
    <!-- Vertical stem -->
    <line x1="35" y1="25" x2="35" y2="75"/>
    <!-- Bowl (semicircle) -->
    <path d="M 35 25 L 50 25 A 15 15 0 0 1 50 55 L 35 55"/>
    <!-- Accent: dot at stem base -->
    <circle cx="35" cy="75" r="3" fill="currentColor" stroke="none"/>
  </g>
</svg>
```

### 3.2 Dot Matrix Letter

**Philosophy**: Form a letter using discrete dots. The letter is suggested rather than explicitly drawn, engaging the viewer's pattern recognition.

**Example: Letter "N" formed by circle dots**

```svg
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <g fill="currentColor">
    <!-- Left column -->
    <circle cx="30" cy="25" r="3"/><circle cx="30" cy="37" r="3"/>
    <circle cx="30" cy="49" r="3"/><circle cx="30" cy="61" r="3"/>
    <circle cx="30" cy="73" r="3"/>
    <!-- Diagonal -->
    <circle cx="38" cy="33" r="3"/><circle cx="46" cy="41" r="3"/>
    <circle cx="54" cy="49" r="3"/><circle cx="62" cy="57" r="3"/>
    <!-- Right column -->
    <circle cx="70" cy="25" r="3"/><circle cx="70" cy="37" r="3"/>
    <circle cx="70" cy="49" r="3"/><circle cx="70" cy="61" r="3"/>
    <circle cx="70" cy="73" r="3"/>
  </g>
</svg>
```

---

## Part 4: Design Quality Checklist

When generating a logo, verify each design against these criteria:

### Visual Quality
- [ ] **Balance**: Visual weight is distributed evenly (or intentionally asymmetric)
- [ ] **Negative space**: Empty space is used deliberately, not accidentally
- [ ] **Scale**: Works at 16x16 (favicon), 64x64 (icon), and 512x512 (full size)
- [ ] **Simplicity**: No element exists without purpose
- [ ] **Distinctiveness**: Not easily confused with existing popular logos

### Technical Quality
- [ ] **viewBox**: Uses `0 0 100 100`
- [ ] **Color**: Uses `currentColor` for fills and strokes
- [ ] **Grouping**: Related elements are grouped with `<g>`
- [ ] **Reusability**: Repeated elements use `<defs>` and `<use>`
- [ ] **Accessibility**: No hardcoded colors

### Conceptual Quality
- [ ] **Relevance**: Design elements connect to product concepts
- [ ] **Storytelling**: The design has an explainable narrative
- [ ] **Uniqueness**: Combination of patterns creates something fresh
- [ ] **Versatility**: Works as app icon, website favicon, social media avatar

---

## Part 5: Generation Strategy for 6+ Variants

When creating logos for a project, ensure diversity by following this allocation:

1. **Pure geometric** (1 variant): Clean shapes, no dots or lines
2. **Dot matrix** (1 variant): One of the dot types (circle, rounded rect, capsule, hexagon)
3. **Line system** (1 variant): Lines, curves, or waves
4. **Mixed: dots + geometry** (1 variant): Dots filling or forming a geometric shape
5. **Mixed: lines + geometry** (1 variant): Lines creating or accenting geometric forms
6. **Node network or layered** (1 variant): Multi-element composition

For each variant, vary these dimensions:
- **Density**: Some sparse, some dense
- **Symmetry**: Some symmetric, some asymmetric
- **Weight**: Some bold/heavy, some light/delicate
- **Complexity**: Some with 2-3 elements, some with 10+

This ensures the user sees genuine variety, not 6 variations of the same idea.

---

## Part 6: SVG Implementation Best Practices

### Coordinate System
- Always use `viewBox="0 0 100 100"` — this makes coordinates intuitive (percentages)
- Center most logos around `(50, 50)` for balanced composition
- Leave at least 10-15 units of padding from edges

### Reusable Definitions
```svg
<defs>
  <!-- Define once, use many times -->
  <circle id="dot" r="3" fill="currentColor"/>
  <clipPath id="circle-bound">
    <circle cx="50" cy="50" r="35"/>
  </clipPath>
</defs>
<!-- Usage -->
<use href="#dot" x="30" y="40"/>
<use href="#dot" x="50" y="40"/>
<use href="#dot" x="70" y="40"/>
```

### Animation (Web Display Only)
```svg
<!-- Subtle rotation -->
<g>
  <animateTransform attributeName="transform" type="rotate" from="0 50 50" to="360 50 50" dur="20s" repeatCount="indefinite"/>
  <!-- ... logo content ... -->
</g>

<!-- Breathing effect (opacity pulse) -->
<circle cx="50" cy="50" r="5">
  <animate attributeName="opacity" values="0.3;1;0.3" dur="3s" repeatCount="indefinite"/>
</circle>
```

### Responsive Color
- Use `fill="currentColor"` and `stroke="currentColor"` throughout
- This allows the parent container to control the logo color via CSS `color` property
- For multi-tone logos, use `opacity` to create shading (0.1-0.4 for backgrounds, 0.6-0.8 for secondary, 1.0 for primary)
