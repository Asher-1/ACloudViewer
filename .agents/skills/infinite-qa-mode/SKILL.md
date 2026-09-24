---
name: infinite-qa-mode
description: Puts the agent into a continuous Q&A loop where every reply must end with AskQuestion offering context-aware options. Use when the user mentions "infinite Q&A", "looping Q&A", or "continuous dialogue".
---

# Infinite Q&A Mode

Automatically appends context-aware quick options at the end of every reply to form a continuous dialogue loop.

## Features

- Every reply ends with an AskQuestion offering 3-5 context-aware options
- Always includes the fixed options `Exit Q&A` and `Other question`
- Normal operations (code edits, analysis, search) remain available during the loop
- The loop ends only when the user actively selects "Exit Q&A"
- The whole loop shares a single request budget until the dialogue ends, saving request quota

## Quick Start

Tell the agent in Cursor:

```
Start infinite Q&A mode
```

After each reply the agent shows an option panel; pick one to continue the dialogue until you exit.

## Agent Behavior Rules

1. **Every reply must end with AskQuestion**, offering:
   - 3-5 context-aware options based on the current discussion (specific, actionable)
   - The fixed option `Exit Q&A`
   - The fixed option `Other question`
2. Normal operations (code edits, search, analysis) remain available during the loop; what gets executed is driven by the Q&A outcome
3. **Never exit the loop on your own** — only end when the user picks "Exit Q&A"
4. When the user picks "Other question", wait for free-form input

## AskQuestion Template

```
AskQuestion:
  title: "Next step"
  questions:
    - id: "next-step"
      prompt: "<prompt based on current discussion>"
      options:
        - { id: "opt1", label: "<option 1>" }
        - { id: "opt2", label: "<option 2>" }
        - { id: "opt3", label: "<option 3>" }
        - { id: "exit-qa", label: "Exit Q&A" }
        - { id: "other", label: "Other question" }
```

Adjust the number and content of options dynamically to the discussion; keep them specific and actionable rather than generic.

## Use Cases

### Use case 1: Debugging

The agent analyzes the problem, edits code, and checks logs while keeping the Q&A loop; each step continues after confirmation.

### Use case 2: Design discussion

Discuss the implementation plan step by step; once confirmed the agent executes the change directly, then the loop continues.

### Use case 3: Code learning

Walk through module logic layer by layer, using options to guide deeper exploration.

## FAQ

**Q: Can code be modified during Q&A mode?**
A: Yes. Q&A mode does not restrict operations; the agent works normally, it just must offer options to continue after each reply.

**Q: How do I exit?**
A: Click the "Exit Q&A" option, or just say "Exit Q&A".

**Q: Will the agent exit by itself?**
A: No. The agent keeps the loop going unless you actively exit.

**Q: How many requests does infinite Q&A mode consume?**
A: The whole loop consumes a single request no matter how many rounds you ask. AskQuestion replies are not new requests — they continue the current dialogue. This saves request quota compared to asking separate questions.
