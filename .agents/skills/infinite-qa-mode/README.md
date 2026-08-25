# Infinite Q&A Mode

Puts the agent into a continuous Q&A loop for scenarios that need repeated discussion and step-by-step confirmation before execution.

## Problem Solved

By default the dialogue ends after each reply and the user must type manually to continue. Infinite Q&A mode makes the agent append context-aware quick options after every reply, forming a continuous interaction loop. The agent can still perform any operation (code edits, search, analysis, etc.) — every step is just guided by options until the user actively exits.

## Installation

Download from the Skill Market and extract to `~/.agents/skills/infinite-qa-mode/`

## Usage

Tell the agent in Cursor:

```
Start infinite Q&A mode
```

The agent enters the Q&A loop and offers options after every reply. Select "Exit Q&A" to end the loop and proceed with execution.

## Project Structure

```
infinite-qa-mode/
├── .skill/
│   └── skill-manifest.json
├── SKILL.md
└── README.md
```

---

Author: @yunshanpeng
