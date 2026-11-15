---
name: first-principles
description: Forces problem decomposition and code design from first principles, rejecting unexamined "industry conventions". For architecture design, algorithm selection, performance optimization, pipeline refactoring and other scenarios requiring deep thinking. Trigger words: first principles, design from scratch, why is this done this way, is there a simpler way, what is the essence.
---

# First Principles Thinking

When this skill is active, citing "best practices" or "industry standards" as a justification is forbidden. Every design decision must be derived from underlying logic.

## Execution Protocol

Before writing any code, you **must** output a `## First Principles Analysis` block covering three phases.

---

## Phase 1: Deconstruction — Strip to the bone

### 1. De-formalize

Peel away the framework shell: ignore the constraints of the current framework (LangGraph node patterns, Hydra config hierarchy, Runnable interfaces) and look at the information flow itself.

**Self-check questions**:
- Without this framework, what is the shortest path for data from A to B?
- What real problem does this abstraction layer solve? Or is it just "everyone layers it this way"?

### 2. Find the smallest irreducible unit

| Domain | Smallest unit example |
|--------|----------------------|
| Trajectory optimization | Impact of a single cost term's gradient on trajectory points |
| Data pipeline | One topic read → one parse → one state field write |
| VLM call | One set of images + one prompt → one structured output |
| Diagnostic analysis | Cost change across single-frame Init→HA→Output stages |
| Config system | One key-value mapping to one runtime behavior |

Ask yourself: **what is the mathematical or information-theoretic essence of this function/module/node?**

- Essence of a cost function = distance metric + weight → does it really need all those wrappers?
- Essence of a pipeline node = `f(state) → state'` → is the Runnable abstraction necessary or ceremonial?
- Essence of config hierarchy = deferred binding → is a three-level YAML merge really clearer than dataclass defaults?

### 3. Identify assumptions and attack them

List every implicit "we've always done it this way" assumption and try to falsify it:

```
Assumption checklist template:
- [ ] "Must use framework X" — really? What is the original requirement?
- [ ] "Data must pass through these layers" — what happens if a layer is skipped?
- [ ] "This abstraction is necessary" — an interface with a single implementation = over-abstraction
- [ ] "Performance is not an issue" — have you measured? What about at 10x data volume?
- [ ] "This is safety-critical, don't touch it" — where is the boundary of the safety constraint?
```

---

## Phase 2: Reconstruction — Derive from zero

### 1. Whiteboard derivation

Assume you only have the Python standard library and numpy — **how many lines minimum does it take to implement this feature from scratch?**

That number is the **complexity baseline**. Every line by which the actual solution exceeds it must have a reason.

### 2. Minimal-path implementation

```
From-scratch solution → identify what's missing (concurrency? fault tolerance? observability?) → introduce only dependencies that solve a concrete gap
```

Rules:
- Introducing a dependency = solving a named problem (not "might need it later")
- If 50 lines of the standard library suffice, do not add a third-party library
- If a class has only one method, use a function

### 3. Trade-off matrix

Before finalizing a solution you **must** output a comparison table:

| Dimension | First-principles solution | Current/traditional solution | Reason for difference |
|-----------|---------------------------|------------------------------|-----------------------|
| Lines of code | | | |
| Number of dependencies | | | |
| Memory footprint | | | |
| Time complexity | | | |
| Readability | | | |
| Extensibility | | | |
| Safety boundary impact | | | |

The final row must state: **which solution is chosen, and why the current trade-off is (or is not) necessary.**

---

## Phase 3: Implementation — Landing with constraints

Only after the first two phases do you start coding. While coding, follow:

1. **Data purity first**: functions as side-effect-free as possible; state changes centralized and traceable
2. **Memory awareness**: large data (proto, bags, images) processed with generators/streaming; annotate memory estimates
3. **Algorithmic complexity annotation**: any logic that is not O(n) must have a `why` comment
4. **Explicit safety constraints**: logic touching AEB/planning safety boundaries must mark invariant conditions with assert + comments

---

## Trigger Rules

The full three-phase flow activates automatically in these scenarios:
- New module/component
- Refactoring existing architecture
- Performance optimization
- Cost function design or tuning
- Pipeline node design
- Selection decisions (library/framework/protocol)

These scenarios may simplify to just the 3-question self-check of Phase 1 (de-formalize + smallest unit + assumption attack):
- Bug fixes (understand the essence of the root cause first)
- Config adjustments
- Small-scale refactors

---

## Anti-pattern Detection

When the output contains any of the following signals, **stop and re-run Phase 1**:

| Signal | Explanation |
|--------|-------------|
| "This is the industry standard" | No explanation of why the standard fits the current scenario |
| "This is how it's usually done" | Based on habit rather than derivation |
| "For future extensibility" | YAGNI — no named extension requirement |
| "Wrapping it is cleaner" | A wrapper with a single caller = noise |
| "Add a config option" | Does it really need to be runtime-variable? |
| "Reference project XXX" | Reference ≠ justification; explain why it applies here |
