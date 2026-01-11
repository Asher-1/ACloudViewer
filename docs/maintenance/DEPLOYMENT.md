# ACloudViewer Website Deployment Guide

## Quick Deployment Steps

### 1. Prerequisites

> Ensure you have:
- Cloned the ACloudViewer repository
- Write permissions to the GitHub repository
- Git properly configured

### 2. Deploy to GitHub Pages

#### Step 1: Push Code to GitHub

```bash
# Navigate to project root
cd /path/to/ACloudViewer

# Add all changes
git add docs/

# Commit changes
git commit -m "Add GitHub Pages website"

# Push to remote repository
git push origin main
```

#### Step 2: Configure GitHub Pages

> Follow these steps to enable GitHub Pages:

1. Visit your GitHub repository: `https://github.com/Asher-1/ACloudViewer`

2. Click the **Settings** tab

3. Find **Pages** in the left sidebar

4. Under **Source** section:
   - **Branch**: Select `main` (or `master`)
   - **Folder**: Select `/docs`
   - Click **Save**

5. Wait for deployment (usually 2-5 minutes)

6. Once deployed, you'll see a green notification:
   ```
   Your site is published at https://asher-1.github.io/ACloudViewer/docs
   ```

### 3. Verify Deployment

> Open your browser and visit:

```
https://asher-1.github.io/ACloudViewer/docs
```

> Check these features:

- ✅ Homepage loads correctly
- ✅ All images display properly
- ✅ Navigation links work
- ✅ Download buttons are functional
- ✅ Mobile responsive design works
- ✅ Search engines can access (robots.txt)

## Detailed Deployment Guide

### File Structure Overview

```
docs/
├── index.html          # Main homepage
├── styles.css          # Stylesheet
├── script.js           # JavaScript
├── .nojekyll          # GitHub Pages config
├── 404.html           # Error page
├── robots.txt         # SEO config
├── sitemap.xml        # Site map
└── images/            # Image assets
```

### Important Files

#### `.nojekyll`

> This file tells GitHub Pages not to process files with Jekyll

**Purpose**:
- Allows files starting with underscore
- Preserves original directory structure
- Faster deployment

**Content**:
```
# Empty file - just needs to exist
```

#### `robots.txt`

> Controls search engine crawler access

**Content**:
```txt
User-agent: *
Allow: /
Sitemap: https://asher-1.github.io/ACloudViewer/docs/sitemap.xml
```

#### `sitemap.xml`

> Helps search engines index your site

**Content**:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://asher-1.github.io/ACloudViewer/docs</loc>
        <lastmod>2026-01-10</lastmod>
        <changefreq>weekly</changefreq>
        <priority>1.0</priority>
    </url>
</urlset>
```

## Deployment Options

### Option 1: GitHub Pages (Recommended)

> **Advantages**:
- ✅ Free hosting
- ✅ Automatic HTTPS
- ✅ CDN distribution
- ✅ Easy setup
- ✅ Custom domain support

> **Limitations**:
- ⚠️ Public repositories only (for free tier)
- ⚠️ 1GB repository size limit
- ⚠️ 100GB bandwidth per month
- ⚠️ 10 builds per hour

> **Best for**: Open source projects, documentation sites

### Option 2: Netlify

> **Setup steps**:

1. Create a Netlify account
2. Connect your GitHub repository
3. Configure build settings:
   - Build command: (none)
   - Publish directory: `docs`
4. Deploy

> **Advantages**:
- ✅ Instant deployments
- ✅ Preview branches
- ✅ Form handling
- ✅ Serverless functions

### Option 3: Vercel

> **Setup steps**:

1. Install Vercel CLI: `npm i -g vercel`
2. Navigate to project: `cd ACloudViewer`
3. Deploy: `vercel --prod`
4. Follow prompts

> **Advantages**:
- ✅ Fast global CDN
- ✅ Automatic HTTPS
- ✅ Preview URLs
- ✅ Analytics included

### Option 4: Custom Server

> **Requirements**:
- Web server (Apache, Nginx)
- SSL certificate
- Domain name (optional)

> **Nginx configuration example**:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    root /path/to/ACloudViewer/docs;
    index index.html;

    location / {
        try_files $uri $uri/ =404;
    }

    # Cache static assets
    location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

## Continuous Deployment

### GitHub Actions Workflow

> Automate deployment with GitHub Actions:

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy Website

on:
  push:
    branches: [main]
    paths:
      - 'docs/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs
```

### Automated Checks

> Add pre-deployment validation:

```yaml
- name: Validate HTML
  run: |
    npm install -g html-validator-cli
    html-validator docs/index.html

- name: Check Links
  run: |
    npm install -g broken-link-checker
    blc https://asher-1.github.io/ACloudViewer/docs -ro
```

## Custom Domain Setup

### Step 1: Configure DNS

> Add DNS records with your domain provider:

```
Type: CNAME
Name: www (or your subdomain)
Value: asher-1.github.io
```

### Step 2: Configure GitHub Pages

> In GitHub repository settings:

1. Go to **Settings** → **Pages**
2. Under **Custom domain**, enter: `www.your-domain.com`
3. Click **Save**
4. Wait for DNS check to complete
5. Enable **Enforce HTTPS** (after DNS propagates)

### Step 3: Verify

> Check domain configuration:

```bash
dig www.your-domain.com +short
# Should return: asher-1.github.io
```

## Troubleshooting

### Common Issues

#### Issue 1: 404 Error

> **Symptoms**: Page not found error

**Solutions**:
1. Check GitHub Pages is enabled
2. Verify source folder is `/docs`
3. Ensure `index.html` exists in `docs/`
4. Wait 2-5 minutes after enabling
5. Clear browser cache

#### Issue 2: Styles Not Loading

> **Symptoms**: Unstyled HTML page

**Solutions**:
1. Check CSS file path: `<link href="styles.css">`
2. Verify `styles.css` exists in same directory
3. Check browser console for 404 errors
4. Ensure no typos in file names
5. Clear browser cache

#### Issue 3: Images Not Displaying

> **Symptoms**: Broken image icons

**Solutions**:
1. Use relative paths: `images/photo.png`
2. Check file names (case-sensitive)
3. Verify images exist in `docs/images/`
4. Check image format (PNG, JPG, GIF)
5. Look for console errors

#### Issue 4: Deployment Failed

> **Symptoms**: GitHub Pages build failed

**Solutions**:
1. Check GitHub Actions logs
2. Verify file structure is correct
3. Look for special characters in file names
4. Ensure `.nojekyll` file exists
5. Check repository settings

### Debugging Tools

> **Browser Developer Tools**:

```
F12 (Windows/Linux) or Cmd+Option+I (Mac)
```

> **Useful tabs**:
- **Console**: JavaScript errors
- **Network**: Failed resource loads
- **Elements**: Inspect HTML/CSS

> **Command Line Tools**:

```bash
# Check HTML validity
html-validator docs/index.html

# Find broken links
broken-link-checker https://asher-1.github.io/ACloudViewer/docs

# Test mobile responsiveness
lighthouse https://asher-1.github.io/ACloudViewer/docs --view
```

## Performance Optimization

### Image Optimization

> **Tools**:
- [TinyPNG](https://tinypng.com/) - Compress PNG/JPG
- [ImageOptim](https://imageoptim.com/) - Mac image optimizer
- [Squoosh](https://squoosh.app/) - Online image compressor

> **Best practices**:
- Compress images before upload
- Use appropriate formats (PNG for graphics, JPG for photos)
- Consider WebP format for modern browsers
- Use responsive images with `srcset`

### Code Minification

> **CSS Minification**:

```bash
npx csso styles.css -o styles.min.css
```

> **JavaScript Minification**:

```bash
npx terser script.js -o script.min.js
```

> **HTML Minification**:

```bash
npx html-minifier docs/index.html -o docs/index.min.html
```

### Caching Strategy

> **Set cache headers** (works on custom servers):

```html
<meta http-equiv="Cache-Control" content="public, max-age=31536000">
```

### CDN Integration

> **Use CDN for libraries**:

```html
<!-- Font Awesome from CDN -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">

<!-- jQuery from CDN (if needed) -->
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
```

## Security Best Practices

### HTTPS Configuration

> GitHub Pages automatically provides HTTPS

- ✅ Automatic SSL certificate
- ✅ Enforced HTTPS option
- ✅ HTTP to HTTPS redirect

### Content Security Policy

> Add CSP headers:

```html
<meta http-equiv="Content-Security-Policy" 
      content="default-src 'self'; script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com;">
```

### Security Headers

> For custom servers, add headers:

```nginx
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "no-referrer-when-downgrade" always;
```

## Monitoring & Analytics

### Google Analytics Setup

```html
<!-- Add before </head> -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

### Uptime Monitoring

> **Recommended services**:
- [UptimeRobot](https://uptimerobot.com/) - Free monitoring
- [Pingdom](https://www.pingdom.com/) - Website monitoring
- [StatusCake](https://www.statuscake.com/) - Uptime monitoring

### Performance Monitoring

> **Tools**:
- [Google PageSpeed Insights](https://pagespeed.web.dev/)
- [GTmetrix](https://gtmetrix.com/)
- [WebPageTest](https://www.webpagetest.org/)

## Backup & Recovery

### Regular Backups

> **Backup checklist**:
- ✅ Git repository (automatic versioning)
- ✅ Local copy of `docs/` folder
- ✅ Image assets backup
- ✅ Configuration files

### Restore from Backup

```bash
# Clone repository
git clone https://github.com/Asher-1/ACloudViewer.git

# Checkout specific version
git checkout <commit-hash>

# Restore docs folder
cp -r docs/ /path/to/restore/
```

## Maintenance Schedule

### Weekly Tasks
- ✅ Check for broken links
- ✅ Review analytics data
- ✅ Monitor uptime status

### Monthly Tasks
- ✅ Update dependencies
- ✅ Review and update content
- ✅ Check SEO performance
- ✅ Test on different browsers

### Quarterly Tasks
- ✅ Major content updates
- ✅ Design improvements
- ✅ Performance optimization
- ✅ Security audit

## Support Resources

> **Official Documentation**:
- [GitHub Pages Docs](https://docs.github.com/en/pages)
- [GitHub Actions Docs](https://docs.github.com/en/actions)

> **Community**:
- [GitHub Community Forum](https://github.community/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/github-pages)

> **Project Support**:
- [Issues](https://github.com/Asher-1/ACloudViewer/issues)
- [Discussions](https://github.com/Asher-1/ACloudViewer/discussions)

---

> **Maintained by**: ACloudViewer Team  
> **Last Updated**: 2026-01-10  
> **Deployment Status**: ✅ Live at https://asher-1.github.io/ACloudViewer/docs
