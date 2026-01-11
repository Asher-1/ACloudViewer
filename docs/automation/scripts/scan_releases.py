#!/usr/bin/env python3
"""
GitHub Releases Scanner and Download Data Generator
自动扫描GitHub Releases并生成下载选择器数据

This script:
1. Fetches all releases from GitHub API
2. Parses asset names to detect platforms, Python versions, architectures, etc.
3. Automatically discovers supported configurations
4. Generates JSON data for the download selector
"""

import os
import re
import json
import ssl
from datetime import datetime
from typing import Dict, List, Optional, Set
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from collections import defaultdict

# GitHub repository information
GITHUB_REPO = "Asher-1/ACloudViewer"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases"
BETA_RELEASE_TAG = "main-devel"

# Regex patterns for parsing asset names

# Pattern 1: Standard format (current)
# ACloudViewer-3.9.3+d236e24-win-cpu-amd64.exe
APP_NAME_PATTERN = re.compile(
    r"ACloudViewer-(?P<version>[\d.]+(?:[+-][\w.]+)?)-"
    r"(?P<platform>win|mac|ubuntu[\d.]+|linux)-"
    r"(?P<cuda>cpu|cuda[\d.]*)-"
    r"(?P<arch>amd64|ARM64|x64|x86)\.(?P<ext>exe|dmg|pkg|run|deb|AppImage)",
    re.IGNORECASE
)

# Pattern 2: Date-based format (legacy)
# ACloudViewer-3.9.2-2024-12-24-win-cpu-amd64.exe
APP_NAME_PATTERN_DATE = re.compile(
    r"(?:ACloudViewer|CloudViewer)-(?P<version>[\d.]+)-(?P<date>\d{4}-\d{2}-\d{2})-"
    r"(?P<platform>win|mac|ubuntu[\d.]+|linux)-"
    r"(?P<cuda>cpu|cuda[\d.]*)-"
    r"(?P<arch>amd64|ARM64|x64|x86)\.(?P<ext>exe|dmg|pkg|run|deb|AppImage)",
    re.IGNORECASE
)

# Pattern 1: Standard wheel format
# cloudviewer-3.9.3+d236e24-cp310-cp310-win_amd64.whl
# cloudviewer-3.9.3+d236e24-cp310-cp310-manylinux_2_35_x86_64.whl
WHEEL_NAME_PATTERN = re.compile(
    r"cloudviewer(?:_cpu)?-(?P<version>[\d.]+(?:[+-][\w.]+)?)-"
    r"cp(?P<python_major>\d)(?P<python_minor>\d+)-"
    r"(?P<abi>.*?)-"
    r"(?P<platform_arch>win_amd64|macosx_[\d_]+_(?:arm64|x86_64)|manylinux_(?P<manylinux_major>\d+)_(?P<manylinux_minor>\d+)_x86_64)\.whl",
    re.IGNORECASE
)

# Pattern 2: Universal2 wheel format (macOS)
# cloudViewer-3.9.3-cp310-cp310-macosx_10_15_universal2.whl
WHEEL_NAME_PATTERN_UNIVERSAL2 = re.compile(
    r"cloudviewer(?:_cpu)?-(?P<version>[\d.]+(?:[+-][\w.]+)?)-"
    r"cp(?P<python_major>\d)(?P<python_minor>\d+)-"
    r"(?P<abi>.*?)-"
    r"(?P<platform_arch>macosx_[\d_]+_universal2)\.whl",
    re.IGNORECASE
)


def get_github_token() -> Optional[str]:
    """Get GitHub Token from environment"""
    return os.environ.get('GITHUB_TOKEN')


def fetch_releases(limit: int = 10) -> List[Dict]:
    """Fetch releases from GitHub API"""
    print(f"📡 Fetching releases from {GITHUB_API_URL}...")
    
    request = Request(GITHUB_API_URL)
    request.add_header('Accept', 'application/vnd.github.v3+json')
    
    token = get_github_token()
    if token:
        request.add_header('Authorization', f'token {token}')
        print("✓ Using GitHub token for authentication")
    
    # Bypass SSL verification for local testing (not recommended for production)
    context = ssl._create_unverified_context()
    
    try:
        with urlopen(request, timeout=30, context=context) as response:
            releases = json.loads(response.read().decode('utf-8'))
        print(f"✅ Found {len(releases)} releases")
        return releases[:limit]
    except HTTPError as e:
        print(f"❌ HTTP Error: {e.code} - {e.reason}")
        raise
    except URLError as e:
        print(f"❌ URL Error: {e.reason}")
        raise


def parse_app_asset(asset_name: str) -> Optional[Dict]:
    """Parse application asset filename"""
    # Try standard format first
    match = APP_NAME_PATTERN.match(asset_name.lower())
    if not match:
        # Try date-based format
        match = APP_NAME_PATTERN_DATE.match(asset_name.lower())
    
    if not match:
        return None
    
    data = match.groupdict()
    
    # Standardize platform
    if 'win' in data['platform']:
        data['platform'] = 'windows'
    elif 'mac' in data['platform'] or 'darwin' in data['platform']:
        data['platform'] = 'macos'
    elif 'ubuntu' in data['platform']:
        data['os_version'] = data['platform']  # Keep ubuntu20.04 format
        data['platform'] = 'linux'
    elif 'linux' in data['platform']:
        data['platform'] = 'linux'
        data['os_version'] = 'ubuntu20.04'  # Default
    
    # Standardize architecture
    if data['arch'] in ['amd64', 'x64', 'x86']:
        data['arch'] = 'amd64'
    elif data['arch'].lower() == 'arm64':
        data['arch'] = 'arm64'
    
    # Standardize CUDA
    if data['cuda'] == 'cpu':
        data['cuda'] = 'cpu'
    elif 'cuda' in data['cuda']:
        data['cuda'] = 'cuda'
    
    data['type'] = 'app'
    return data


def parse_wheel_asset(asset_name: str) -> Optional[Dict]:
    """Parse Python wheel asset filename"""
    # Try standard format first
    match = WHEEL_NAME_PATTERN.match(asset_name.lower())
    if not match:
        # Try universal2 format
        match = WHEEL_NAME_PATTERN_UNIVERSAL2.match(asset_name.lower())
    
    if not match:
        return None
    
    data = match.groupdict()
    data['python_version'] = f"{data['python_major']}.{data['python_minor']}"
    
    # Determine if it's CPU-only wheel
    data['cuda'] = 'cpu' if 'cpu' in asset_name.lower() else 'cuda'
    
    # Parse platform and architecture
    platform_arch = data['platform_arch']
    if 'win_amd64' in platform_arch:
        data['platform'] = 'windows'
        data['arch'] = 'amd64'
    elif 'macosx' in platform_arch:
        data['platform'] = 'macos'
        if 'universal2' in platform_arch:
            # Universal2 supports both arm64 and x86_64
            # We'll create entries for both architectures
            data['arch'] = 'universal2'  # Special marker
        elif 'arm64' in platform_arch:
            data['arch'] = 'arm64'
        else:
            data['arch'] = 'amd64'
    elif 'manylinux' in platform_arch:
        data['platform'] = 'linux'
        data['arch'] = 'amd64'
        
        # Map manylinux version to Ubuntu version
        # manylinux_2_27 = Ubuntu 18.04
        # manylinux_2_31 = Ubuntu 20.04
        # manylinux_2_35 = Ubuntu 22.04
        # manylinux_2_39 = Ubuntu 24.04
        manylinux_minor = data.get('manylinux_minor', '35')
        manylinux_map = {
            '27': 'ubuntu18.04',
            '31': 'ubuntu20.04',
            '35': 'ubuntu22.04',
            '39': 'ubuntu24.04'
        }
        data['os_version'] = manylinux_map.get(manylinux_minor, 'ubuntu22.04')
        print(f"   ℹ️  Mapped manylinux_2_{manylinux_minor} → {data['os_version']}")
    
    data['type'] = 'wheel'
    return data


def analyze_releases(releases: List[Dict]) -> Dict:
    """Analyze all releases and extract supported configurations"""
    print("\n🔍 Analyzing releases...")
    
    version_info = {}
    
    for release in releases:
        tag = release['tag_name']
        display_name = "Beta" if tag == BETA_RELEASE_TAG else tag
        
        print(f"\n📦 Processing {display_name} ({len(release['assets'])} assets)")
        
        # Track what this version supports
        supported_platforms = set()
        supported_pythons = set()
        supported_ubuntu = set()
        assets_data = []
        
        for asset in release['assets']:
            name = asset['name']
            
            # Try parsing as app
            parsed = parse_app_asset(name)
            if not parsed:
                # Try parsing as wheel
                parsed = parse_wheel_asset(name)
            
            if parsed:
                asset_info = {
                    'name': name,
                    'url': asset['browser_download_url'],
                    'size': round(asset['size'] / (1024 * 1024), 1),  # MB
                    **parsed
                }
                assets_data.append(asset_info)
                
                # Track capabilities
                supported_platforms.add(parsed['platform'])
                if parsed['type'] == 'wheel':
                    supported_pythons.add(parsed['python_version'])
                if parsed.get('os_version') and 'ubuntu' in parsed.get('os_version', ''):
                    supported_ubuntu.add(parsed['os_version'])
            else:
                print(f"  ⚠️  Could not parse: {name}")
        
        version_info[tag] = {
            'display_name': display_name,
            'tag': tag,
            'published_at': release['published_at'],
            'is_prerelease': release.get('prerelease', False),
            'platforms': sorted(list(supported_platforms)),
            'python_versions': sorted(list(supported_pythons)),
            'ubuntu_versions': sorted(list(supported_ubuntu)),
            'assets': assets_data
        }
        
        print(f"  ✓ Platforms: {', '.join(sorted(supported_platforms))}")
        print(f"  ✓ Python: {', '.join(sorted(supported_pythons)) or 'N/A'}")
        print(f"  ✓ Ubuntu: {', '.join(sorted(supported_ubuntu)) or 'N/A (wheel uses manylinux)'}")
        print(f"  ✓ Parsed {len(assets_data)} / {len(release['assets'])} assets")
    
    return version_info


def build_download_structure(version_info: Dict) -> Dict:
    """Build the nested download data structure"""
    print("\n🏗️  Building download data structure...")
    
    download_data = {}
    
    for tag, info in version_info.items():
        version_data = {}
        
        for asset in info['assets']:
            platform = asset['platform']
            asset_type = asset['type']
            cuda = asset['cuda']
            arch = asset['arch']
            
            # Initialize nested structure
            if platform not in version_data:
                version_data[platform] = {}
            
            if asset_type == 'app':
                # For apps, organize by os_version (for Linux) or directly
                if platform == 'linux' and asset.get('os_version'):
                    os_version = asset['os_version']
                    if os_version not in version_data[platform]:
                        version_data[platform][os_version] = {'app': {}, 'wheel': {}}
                    
                    if 'app' not in version_data[platform][os_version]:
                        version_data[platform][os_version]['app'] = {}
                    if cuda not in version_data[platform][os_version]['app']:
                        version_data[platform][os_version]['app'][cuda] = {}
                    
                    version_data[platform][os_version]['app'][cuda][arch] = {
                        'url': asset['url'],
                        'size': f"{asset['size']} MB"
                    }
                else:
                    # Non-Linux apps
                    if 'app' not in version_data[platform]:
                        version_data[platform]['app'] = {}
                    if cuda not in version_data[platform]['app']:
                        version_data[platform]['app'][cuda] = {}
                    
                    version_data[platform]['app'][cuda][arch] = {
                        'url': asset['url'],
                        'size': f"{asset['size']} MB"
                    }
            
            elif asset_type == 'wheel':
                python_ver = asset['python_version']
                
                # For Linux wheels, organize by Ubuntu version (based on manylinux version)
                if platform == 'linux' and asset.get('os_version'):
                    os_version = asset['os_version']
                    if os_version not in version_data[platform]:
                        version_data[platform][os_version] = {'app': {}, 'wheel': {}}
                    
                    if 'wheel' not in version_data[platform][os_version]:
                        version_data[platform][os_version]['wheel'] = {}
                    if cuda not in version_data[platform][os_version]['wheel']:
                        version_data[platform][os_version]['wheel'][cuda] = {}
                    
                    if arch not in version_data[platform][os_version]['wheel'][cuda]:
                        version_data[platform][os_version]['wheel'][cuda][arch] = {}
                    
                    version_data[platform][os_version]['wheel'][cuda][arch][python_ver] = {
                        'url': asset['url'],
                        'size': f"{asset['size']} MB"
                    }
                else:
                    # Non-Linux wheels (Windows, macOS)
                    if 'wheel' not in version_data[platform]:
                        version_data[platform]['wheel'] = {}
                    if cuda not in version_data[platform]['wheel']:
                        version_data[platform]['wheel'][cuda] = {}
                    
                    # Handle universal2 wheels (create entries for both arm64 and amd64)
                    if arch == 'universal2':
                        for target_arch in ['arm64', 'amd64']:
                            if target_arch not in version_data[platform]['wheel'][cuda]:
                                version_data[platform]['wheel'][cuda][target_arch] = {}
                            version_data[platform]['wheel'][cuda][target_arch][python_ver] = {
                                'url': asset['url'],
                                'size': f"{asset['size']} MB"
                            }
                    else:
                        if arch not in version_data[platform]['wheel'][cuda]:
                            version_data[platform]['wheel'][cuda][arch] = {}
                        
                        version_data[platform]['wheel'][cuda][arch][python_ver] = {
                            'url': asset['url'],
                            'size': f"{asset['size']} MB"
                        }
        
        download_data[tag] = version_data
    
    return download_data


def generate_version_metadata(version_info: Dict) -> List[Dict]:
    """Generate metadata for version selector buttons"""
    metadata = []
    
    for tag, info in version_info.items():
        # Collect all unique Python versions across all assets
        python_versions = set()
        ubuntu_versions = set()
        
        for asset in info['assets']:
            if asset['type'] == 'wheel' and asset.get('python_version'):
                python_versions.add(asset['python_version'])
            if asset.get('os_version') and 'ubuntu' in asset.get('os_version', ''):
                ubuntu_versions.add(asset['os_version'])
        
        metadata.append({
            'value': tag,
            'display_name': info['display_name'],
            'python_versions': sorted(list(python_versions)),
            'ubuntu_versions': sorted(list(ubuntu_versions)),
            'is_default': tag == BETA_RELEASE_TAG
        })
    
    return metadata


def main():
    print("=" * 80)
    print("🚀 GitHub Releases Scanner")
    print("=" * 80)
    
    try:
        # Fetch releases
        releases = fetch_releases(limit=5)  # Get latest 5 releases
        
        # Analyze releases
        version_info = analyze_releases(releases)
        
        # Build download structure
        download_data = build_download_structure(version_info)
        
        # Generate version metadata
        version_metadata = generate_version_metadata(version_info)
        
        # Prepare output
        output = {
            'generated_at': datetime.now().isoformat(),
            'version_metadata': version_metadata,
            'download_data': download_data
        }
        
        # Save to JSON file
        output_file = 'docs/downloads_data.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 80)
        print(f"✅ Successfully generated {output_file}")
        print(f"📊 Summary:")
        print(f"   • Versions: {len(version_info)}")
        print(f"   • Total configurations: {sum(len(v['assets']) for v in version_info.values())}")
        print("=" * 80)
        
        # Print sample
        print("\n📝 Sample data structure:")
        print(json.dumps(output['version_metadata'], indent=2))
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

