#!/usr/bin/env python3
"""
自动更新网站下载链接的脚本
从GitHub Releases API获取最新版本信息并更新doc/index.html
"""

import os
import re
import json
import ssl
from datetime import datetime
from typing import Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# GitHub仓库信息
GITHUB_REPO = "Asher-1/ACloudViewer"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases"

# 平台和文件后缀映射
PLATFORM_PATTERNS = {
    'windows': {
        'patterns': [r'windows.*\.exe$', r'win.*\.exe$', r'\.exe$'],
        'display_name': 'Windows'
    },
    'macos': {
        'patterns': [r'macos.*\.(dmg|pkg)$', r'darwin.*\.(dmg|pkg)$', r'osx.*\.(dmg|pkg)$', r'\.(dmg|pkg)$'],
        'display_name': 'macOS'
    },
    'linux': {
        'patterns': [r'linux.*\.run$', r'\.run$', r'linux.*\.(deb|rpm|appimage)$', r'\.(deb|rpm|appimage)$'],
        'display_name': 'Linux'
    },
    'ubuntu': {
        'patterns': [r'ubuntu.*\.deb$', r'\.deb$'],
        'display_name': 'Ubuntu/Debian'
    }
}


def get_github_token() -> Optional[str]:
    """获取GitHub Token"""
    return os.environ.get('GITHUB_TOKEN')


def fetch_releases() -> List[Dict]:
    """从GitHub API获取所有releases"""
    print(f"Fetching releases from {GITHUB_API_URL}...")
    
    request = Request(GITHUB_API_URL)
    request.add_header('Accept', 'application/vnd.github.v3+json')
    
    token = get_github_token()
    if token:
        request.add_header('Authorization', f'token {token}')
    
    # Create SSL context (for local testing without proper cert, use unverified context)
    # In production (GitHub Actions), this won't be needed
    try:
        context = ssl.create_default_context()
    except:
        context = ssl._create_unverified_context()
    
    try:
        with urlopen(request, timeout=30, context=context) as response:
            releases = json.loads(response.read().decode('utf-8'))
        print(f"Found {len(releases)} releases")
        return releases
    except HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason}")
        raise
    except URLError as e:
        print(f"URL Error: {e.reason}")
        raise


def find_asset_for_platform(assets: List[Dict], platform: str) -> Optional[Dict]:
    """为指定平台查找匹配的asset"""
    if platform not in PLATFORM_PATTERNS:
        return None
    
    patterns = PLATFORM_PATTERNS[platform]['patterns']
    
    for asset in assets:
        name = asset['name'].lower()
        for pattern in patterns:
            if re.search(pattern, name, re.IGNORECASE):
                return asset
    
    return None


def get_beta_release(releases: List[Dict]) -> Optional[Dict]:
    """获取Beta版本（main-devel tag）"""
    for release in releases:
        if release['tag_name'] == 'main-devel':
            print(f"Found beta release: {release['tag_name']}")
            return release
    return None


def get_stable_releases(releases: List[Dict], limit: int = 3) -> List[Dict]:
    """获取稳定版本（非pre-release，排除main-devel）"""
    stable = []
    for release in releases:
        if not release['prerelease'] and release['tag_name'] != 'main-devel':
            stable.append(release)
            if len(stable) >= limit:
                break
    
    print(f"Found {len(stable)} stable releases")
    return stable


def generate_beta_section(beta_release: Dict) -> str:
    """生成Beta版本区域的HTML"""
    commit_sha = beta_release.get('target_commitish', '')[:7]
    version_name = beta_release.get('name', 'main-devel')
    published_date = datetime.strptime(
        beta_release['published_at'], 
        '%Y-%m-%dT%H:%M:%SZ'
    ).strftime('%Y年%m月%d日')
    
    html_parts = [f'''            <!-- Beta Version -->
            <div class="version-section">
                <div class="version-header">
                    <span class="version-badge beta">🚀 Beta 版本: {version_name}</span>
                    <span class="release-date">发布日期: {published_date} | Commit: {commit_sha}</span>
                </div>
                <p class="version-description">最新的测试版本，包含最新功能和改进</p>
                
                <div class="download-grid">''']
    
    # 为每个平台生成下载卡片
    for platform_key, platform_info in PLATFORM_PATTERNS.items():
        asset = find_asset_for_platform(beta_release['assets'], platform_key)
        if asset:
            download_url = asset['browser_download_url']
            file_name = asset['name']
            
            # 确定图标
            icon = 'fab fa-windows' if platform_key == 'windows' else \
                   'fab fa-apple' if platform_key == 'macos' else 'fab fa-linux'
            
            html_parts.append(f'''                    <!-- {platform_info['display_name']} Beta -->
                    <div class="download-card compact">
                        <div class="os-icon-small">
                            <i class="{icon}"></i>
                        </div>
                        <h4>{platform_info['display_name']}</h4>
                        <div class="download-buttons">
                            <a href="{download_url}" class="btn btn-download-small">
                                <i class="fas fa-download"></i> 下载
                            </a>
                        </div>
                    </div>
''')
    
    html_parts.append('''                </div>
            </div>
''')
    
    return '\n'.join(html_parts)


def generate_stable_section(stable_releases: List[Dict]) -> str:
    """生成稳定版本区域的HTML"""
    html_parts = ['''            <!-- Stable Releases -->
            <div class="version-section stable">
                <div class="version-header">
                    <span class="version-badge stable">✅ 稳定版本</span>
                </div>
                <p class="version-description">经过测试的稳定版本，推荐用于生产环境</p>
                
                <!-- Version Tabs -->
                <div class="version-tabs">''']
    
    # 生成版本标签
    for idx, release in enumerate(stable_releases):
        version = release['tag_name']
        version_number = version.replace('v', '')
        active_class = ' active' if idx == 0 else ''
        html_parts.append(f'                    <button class="version-tab{active_class}" data-version="{version_number}">{version}</button>')
    
    html_parts.append('                </div>\n')
    
    # 生成每个版本的下载内容
    for idx, release in enumerate(stable_releases):
        version = release['tag_name']
        version_number = version.replace('v', '')
        active_class = ' active' if idx == 0 else ''
        
        html_parts.append(f'''                <!-- {version} Downloads -->
                <div class="version-content{active_class}" id="version-{version_number}">
                    <div class="download-grid">''')
        
        # 为每个平台生成下载卡片
        for platform_key, platform_info in PLATFORM_PATTERNS.items():
            asset = find_asset_for_platform(release['assets'], platform_key)
            if asset:
                download_url = asset['browser_download_url']
                
                # 确定图标
                icon = 'fab fa-windows' if platform_key == 'windows' else \
                       'fab fa-apple' if platform_key == 'macos' else 'fab fa-linux'
                
                html_parts.append(f'''                        <div class="download-card compact">
                            <div class="os-icon-small"><i class="{icon}"></i></div>
                            <h4>{platform_info['display_name']}</h4>
                            <div class="download-buttons">
                                <a href="{download_url}" class="btn btn-download-small">
                                    <i class="fas fa-download"></i> 下载
                                </a>
                            </div>
                        </div>''')
        
        html_parts.append('''                    </div>
                </div>
''')
    
    html_parts.append('            </div>')
    
    return '\n'.join(html_parts)




def update_html_file(beta_release: Optional[Dict], stable_releases: List[Dict]):
    """更新HTML文件"""
    html_file = 'docs/index.html'
    
    if not os.path.exists(html_file):
        print(f"Error: {html_file} not found!")
        return
    
    print(f"Reading {html_file}...")
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 生成新的内容
    new_html_parts = []
    
    if beta_release:
        new_html_parts.append(generate_beta_section(beta_release))
    
    if stable_releases:
        new_html_parts.append(generate_stable_section(stable_releases))
    
    new_download_html = '\n'.join(new_html_parts)
    
    # 查找并替换整个下载区域
    # 从第一个 <!-- Beta Version --> 或 <div class="version-section"> 
    # 到 <div class="all-releases"> 之前
    download_pattern = r'(            <!-- Beta Version -->.*?)(            <div class="all-releases">)'
    
    replacement = new_download_html + '\n\n' + r'\2'
    
    content = re.sub(
        download_pattern,
        replacement,
        content,
        flags=re.DOTALL
    )
    
    print(f"Writing updated content to {html_file}...")
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Successfully updated download links!")


def main():
    """主函数"""
    try:
        print("=" * 60)
        print("Starting download links update process...")
        print("=" * 60)
        
        # 获取releases
        releases = fetch_releases()
        
        if not releases:
            print("No releases found!")
            return
        
        # 获取Beta和稳定版本
        beta_release = get_beta_release(releases)
        stable_releases = get_stable_releases(releases, limit=3)
        
        if not beta_release and not stable_releases:
            print("No beta or stable releases found!")
            return
        
        # 更新HTML文件
        update_html_file(beta_release, stable_releases)
        
        print("=" * 60)
        print("Update process completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()

