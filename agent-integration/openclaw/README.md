# ACloudViewer OpenClaw Integration

Use ACloudViewer as an OpenClaw skill through the Model Context Protocol.

## Setup

### Option 1: Managed (OpenClaw Launch)

Search for "acloudviewer" in the ClawHub marketplace and toggle it on.

### Option 2: Self-Hosted

1. Install the CLI package:

```bash
pip install 'cli-anything-acloudviewer[mcp]'
```

2. Add to your `openclaw.json`:

```json
{
  "plugins": {
    "acloudviewer": {
      "command": "cli-anything-acloudviewer-mcp",
      "type": "mcp",
      "description": "3D point cloud and mesh processing with ACloudViewer"
    }
  }
}
```

3. Restart OpenClaw.

## Usage Examples

Once configured, ask OpenClaw to:

- "Load scene.ply and show me the point count"
- "Downsample the point cloud with voxel size 0.05"
- "Register source.ply to target.ply using ICP"
- "Remove outliers from noisy_scan.ply"
- "Reconstruct a mesh from the point cloud using Poisson"
- "Take a screenshot of the current viewport"

## Available Tools

See [../mcp/README.md](../mcp/README.md) for the full list of 23 MCP tools.

## Skill Manifest

The `openclaw-skill.json` file in this directory provides the skill manifest
for ClawHub registration.
