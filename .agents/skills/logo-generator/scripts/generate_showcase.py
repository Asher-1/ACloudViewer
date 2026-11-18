#!/usr/bin/env python3
"""
Generate logo showcase images using Nano Banana (Gemini Image Generation API).
Supports both official Google API and third-party API endpoints.
"""

import os
import sys
import base64
import argparse
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: google-genai package not installed.")
    print("Install with: pip install google-genai")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_BASE_URL = os.getenv("GEMINI_API_BASE_URL", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-image-preview")

# Background style prompts from the reference document
BACKGROUND_STYLES = {
    "void": """THE VOID (绝对虚空)
Absolute black (#000000) background with extremely fine silver/white high-contrast micro noise.
Cold, sharp electronic film grain texture. Minimal atmosphere light - only a faint, icy white or blue glow
at the extreme corner, like distant starlight at the edge of the universe.""",

    "frosted": """FROSTED HORIZON (磨砂穹顶)
Deep titanium gray or midnight slate gray base, not pure black. Organic film-like dust noise texture,
resembling unpolished rough metal or stone surface. Large area but extremely low saturation cold-toned
light halo (low-saturation gray-blue), edges completely dissolved like mist.""",

    "fluid": """FLUID ABYSS (流体深渊)
Deep midnight purple or extremely dark Klein blue base. Noise texture with slight color tint,
blending with the base to create deep-sea sediment or nebula texture. Fluid fusion light -
dark orange on right side, dark blue on left side, slowly interweaving in the dark space center.""",

    "spotlight": """STUDIO SPOTLIGHT (物理影棚)
Extremely dark warm carbon gray base. Slightly larger grain simulating low-light camera photography,
like paper print grain in weak light. Single-side softbox or spotlight creating natural vignette,
editorial magazine quality with professional photography feel.""",

    "analog_liquid": """ANALOG LIQUID (物理流体)
Solid color base - choose ONE color only: vibrant orange (#FF6B00), Klein blue (#002FA7), or lime green (#00FF41).
Physical liquid textures with metallic shimmer overlaying the solid base - gold dust flow, metallic mica powder
suspended in liquid, iridescent pigments creating rainbow oil slick effects. Dry mineral textures like crushed
gemstones. Macro photography of natural materials - copper oxidation, rust patterns, gold leaf fragments.
Extreme grainy texture like thermal imaging or pushed film grain. Create maximum contrast between chaotic
organic texture and ultra-clean sharp vector logo.""",

    "led_matrix": """LED MATRIX (数字硬件)
Black background with glowing dot matrix patterns creating waves of light. Simulate old-school CRT displays,
LED billboards, or halftone printing dots. Retro computer display artifacts with modern execution.
Waves of glowing points creating depth, logo as solid entity floating above. Cyberpunk and retro-futurism
aesthetics with hardcore geek appeal.""",

    "editorial": """EDITORIAL PAPER (纸本编辑)
Off-white, alabaster, or pearl white base (not pure white). High-grade watercolor or rough art paper
texture suggesting physical paper tactile quality. Natural light diffuse reflection with slight warm
gray vignette in corners. Humanistic, independent magazine aesthetic.""",

    "iridescent": """IRIDESCENT FROST (幻彩透砂)
Extremely light silver-gray or cold white base, creating calm, rational experimental space.
Extremely fine micro noise, simulating high-density frosted glass or sandblasted aluminum surface.
Restrained holographic/iridescent atmosphere light - faint low-saturation light purple, light blue
or soft pink fluid diffused light in the clean background depth, like through thick frosted glass.""",

    "morning": """MORNING AURA (晨雾光域)
Warm ivory or extremely light cream color base. Soft noise blending into base like morning mist or dust,
creating thin layer of atmospheric haze. Large area blurred low-saturation pastel colors (mint green,
baby blue, dawn orange) dissolving into warm white. Warm, intelligent, pressure-free atmosphere.""",

    "clinical": """CLINICAL STUDIO (无菌影棚)
Pure white or extremely light cold gray base. High-frequency sharp cold-toned digital micro noise with
enhanced sharpness. Pure light/shadow structure - large softbox from top/side creating smooth gray-white
gradient. Sterile space with geometric order, creating 3D depth in 2D presentation.""",

    "ui_container": """UI CONTAINER (容器化界面)
Clean gradient or solid color background with minimal digital noise. Frosted glass container effect
(like app icon base) with rounded corners and subtle transparency. Micro-shadows creating depth illusion.
UI-native presentation suggesting interactivity and digital product context. Logo sits in transparent
container with modern interface design language.""",

    "swiss_flat": """SWISS FLAT (瑞士扁平)
100% pure solid color background - deep vintage green, rich burgundy, or classic navy. Absolutely no
gradients, no noise, no effects. Pure graphic design with zero tricks. Just perfect color and form.
Extreme confidence and timeless authority. Classic Swiss International Style with absolute flatness."""
}


def load_reference_image(image_path: str) -> Optional[str]:
    """Load and encode reference image to base64."""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        print(f"Error loading reference image: {e}")
        return None


def generate_showcase_image(
    logo_name: str,
    reference_image_path: str,
    style: str,
    output_path: str,
    product_description: str = ""
) -> bool:
    """
    Generate a showcase image using Nano Banana API.

    Args:
        logo_name: Name of the logo/product
        reference_image_path: Path to the reference logo image (SVG exported as PNG)
        style: Background style key (void, frosted, fluid, iridescent)
        output_path: Path to save the generated image
        product_description: Optional product description for context

    Returns:
        True if successful, False otherwise
    """
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not set in environment")
        return False

    if style not in BACKGROUND_STYLES:
        print(f"Error: Unknown style '{style}'. Available: {list(BACKGROUND_STYLES.keys())}")
        return False

    # Load reference image
    reference_image_b64 = load_reference_image(reference_image_path)
    if not reference_image_b64:
        return False

    # Build the prompt
    style_description = BACKGROUND_STYLES[style]

    # Determine logo color based on background style
    # Dark backgrounds use white logo, light backgrounds use black logo
    dark_styles = ["void", "frosted", "fluid", "spotlight", "analog_liquid", "led_matrix"]
    light_styles = ["editorial", "iridescent", "morning", "clinical", "ui_container", "swiss_flat"]

    is_dark_bg = style in dark_styles
    logo_color = "pure white (#FFFFFF)" if is_dark_bg else "pure black (#000000)"

    prompt = f"""Extract the core graphic from the reference image as a pure flat single-color vector structure,
removing all decorations. Use high-contrast atmosphere background, delicate film grain noise,
and rigorous micro-typography to create a cutting-edge, restrained, and highly digital order showcase effect.

LOGO PROCESSING:
- Strip background and outer frames
- Extract core graphic only, preserve graphic details
- Extremely flat: 100% solid color flat vector in {logo_color}
- Sharp, clear edges
- The logo MUST be rendered in {logo_color} to ensure maximum contrast with the background

BACKGROUND CONSTRUCTION:
{style_description}

TYPOGRAPHY AND LAYOUT:
Use classic Swiss-style typography logic with extreme proportion contrast.

- Main subject centered: Place the pure flat logo graphic at the absolute visual center with huge breathing space
- Micro-typography: Remove any large, obtrusive titles. Use extremely small font size (6pt to 9pt)
  and clean sans-serif fonts (Inter, Helvetica, Geist) in corners or bottom center
- Text content suggestions (strictly aligned):
  Left corner: {logo_name.upper()}
  Right corner: v. 1.0.0 // 2026
  Bottom center: {product_description.upper() if product_description else 'DIGITAL IDENTITY SYSTEM'}

CRITICAL: The logo graphic MUST be {logo_color}, perfectly centered, extracted from the reference image,
rendered as pure flat vector with sharp edges."""

    try:
        # Initialize client
        client_config = {"api_key": GEMINI_API_KEY}
        if GEMINI_API_BASE_URL:
            client_config["http_options"] = {"api_endpoint": GEMINI_API_BASE_URL}

        client = genai.Client(**client_config)

        # Prepare content with reference image
        contents = [
            types.Part.from_bytes(
                data=base64.b64decode(reference_image_b64),
                mime_type="image/png"
            ),
            types.Part.from_text(text=prompt)
        ]

        print(f"Generating showcase image with style: {style}")
        print(f"Using model: {GEMINI_MODEL}")
        if GEMINI_API_BASE_URL:
            print(f"Using custom API endpoint: {GEMINI_API_BASE_URL}")

        # Generate image with 16:9 aspect ratio and 2K resolution
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio="16:9",
                    image_size="2K"
                )
            )
        )

        # Save generated image
        for part in response.parts:
            if part.inline_data is not None:
                image = part.as_image()
                image.save(output_path)
                print(f"✓ Showcase image saved: {output_path}")
                return True
            elif part.text is not None:
                print(f"Model response: {part.text}")

        print("Error: No image generated in response")
        return False

    except Exception as e:
        print(f"Error generating showcase image: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate logo showcase images using Nano Banana API"
    )
    parser.add_argument("logo_name", help="Name of the logo/product")
    parser.add_argument("reference_image", help="Path to reference logo image (PNG)")
    parser.add_argument("--style",
                       choices=list(BACKGROUND_STYLES.keys()),
                       default="iridescent",
                       help="Background style")
    parser.add_argument("--output", "-o",
                       help="Output path (default: output/{logo_name}_{style}.png)")
    parser.add_argument("--description", "-d",
                       default="",
                       help="Product description for context")
    parser.add_argument("--all-styles",
                       action="store_true",
                       help="Generate all 13 styles")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    if args.all_styles:
        # Generate all 4 styles
        success_count = 0
        for style in BACKGROUND_STYLES.keys():
            output_path = output_dir / f"{args.logo_name}_{style}.png"
            if generate_showcase_image(
                args.logo_name,
                args.reference_image,
                style,
                str(output_path),
                args.description
            ):
                success_count += 1

        print(f"\n✓ Generated {success_count}/{len(BACKGROUND_STYLES)} showcase images")
    else:
        # Generate single style
        if args.output:
            output_path = args.output
        else:
            output_path = output_dir / f"{args.logo_name}_{args.style}.png"

        success = generate_showcase_image(
            args.logo_name,
            args.reference_image,
            args.style,
            str(output_path),
            args.description
        )

        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
