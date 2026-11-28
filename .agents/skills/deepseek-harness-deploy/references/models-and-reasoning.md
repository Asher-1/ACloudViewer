# Gateway Models, Dual-Key Entitlement, and Reasoning Levels

Verified against token-wormhole.deeproute.cn (2026-11, key-group entitlements
as observed; re-probe when the catalog changes). Never print or paste key
values while following this procedure.

## 1. Gateway Endpoints & Dual Protocol

| Endpoint | Protocol | Serves |
|---|---|---|
| `GET /v1/models` | OpenAI | catalog listing (entitled to the key's group; listed ≠ usable — always probe) |
| `POST /v1/responses` | OpenAI Responses | GPT / Grok / DeepSeek / Qwen / GLM / Kimi chat models |
| `POST /v1/chat/completions` | OpenAI Completions | same pool via chat API |
| `POST /v1/messages` | Anthropic Messages | **Claude models — never listed in `/v1/models`**; auth via `x-api-key` or `Authorization: Bearer` |

Both `Authorization: Bearer` and `x-api-key` are accepted on the Anthropic
entry; DSH's `anthropic-messages` provider sends `x-api-key`.

## 2. Key-Group Entitlement (why Claude needs a second key)

The gateway authorizes models per key group. Two common internal keys coexist
on one machine:

- `~/.codex/auth.json` `OPENAI_API_KEY` — group with GPT/Grok/domestic models,
  **without** Claude (`"Model ... is not available for this group"` on
  `/v1/messages`).
- `~/.claude/settings.json` `env.ANTHROPIC_AUTH_TOKEN` — group **with**
  Claude (claude-sonnet-5-5 / opus-4-6 / haiku-4-5 verified 200).

Discovery procedure:

1. `GET /v1/models` with each candidate key → compare catalogs.
2. `POST /v1/messages` with a known Claude id (`claude-sonnet-5-5`) to test
   the Anthropic group.
3. If two groups are needed, declare two providers in `cordis.patch.yml`
   (e.g. `deeproute-gateway` with `openai-responses`, `deeproute-claude` with
   `anthropic-messages`, same `baseURL`, different `apiKeyEnv`), and store both
   values in `~/.dsh/.credentials.yaml` `refs:`.

Probe a chat model before adding it to `models:`:

```python
# POST /v1/responses  {"model": m, "input": "ping", "max_output_tokens": 128,
#                      "reasoning": {"effort": "high"}}
# 200 = usable; 400 model_not_found / 502 upstream = listed but not served
```

Observed (2026-11): GPT 5.2→6.2 chat ids all usable; grok-4.5/4.6/4.7 usable
(occasional upstream 429); of 26 Gemini ids only `gemini-3-flash` was served
(others 400/502); no claude-* ids in `/v1/models` at all. Exclude image models
(`gpt-image-*`), protocol duplicates (`*-openai-compact`), and placeholders
(`gpt-reserve`) from chat provider lists.

## 3. Reasoning-Effort Declaration (the Effort menu)

The model picker shows an **Effort** menu only for models that declare levels.
Hand-entered models declare nothing → no menu → the endpoint default applies.
Declare per model in `cordis.patch.yml`:

```yaml
        models:
          - id: gpt-6.1-sol
            name: gpt-6.1-sol
            reasoningEfforts:
              off:                 # empty = send no reasoning field (endpoint default)
              minimal: minimal
              low: low
              medium: medium
              high: high
              xhigh: xhigh
```

Rules (validated against `@deepseek-ai/dsh-llm-pi-ai`):

- Level keys (escalation order): `off minimal low medium high xhigh max`.
- The value is the **wire spelling** and must itself be one of those level
  names (`max: xhigh` renames a level for gateways with their own vocabulary).
- Only `off` may be empty (sends nothing). `xhigh` and `max` appear in the menu
  **only if explicitly declared**; `off/minimal/low/medium/high` are offered
  once any declaration exists.
- `reasoningEfforts: false` marks a non-reasoning model; an empty dict is a
  config error.

### Wire dispatch per protocol

| api | effort goes out as | notes |
|---|---|---|
| `openai-responses` | `reasoning: {effort: <value>}` | omitting the field = endpoint default |
| `openai-completions` | `reasoning_effort: <value>` | compat flags may be needed (`compat.supportsDeveloperRole: false` etc.) |
| `anthropic-messages` | adaptive thinking: `thinking: {type: "adaptive"}` + `output_config: {effort: <value>}` | pi-ai maps `minimal/low→low`, `medium→medium`, `high→high`; a declared value is passed through verbatim |

Anthropic specifics (pi-ai `anthropic-messages.js`):

- effort `max` is available on **all** adaptive-thinking Claude models; native
  `xhigh` only on Opus 4.7/4.8, Sonnet 5, Fable 5 (2026-11).
- Older budget-style `thinking: {type: "enabled", budget_tokens: N}` is
  rejected by adaptive-only models on this gateway ("requires adaptive
  thinking or thinking.type=between_tools") — declare levels, let pi-ai send
  adaptive.
- DeepSeek-style models behind OpenAI-compatible gateways that think by
  default need `compat.thinkingFormat: deepseek` so `off` actually stops
  thinking (see providers.md).

### Probe before declaring

Every declared value becomes a request field — an unsupported value turns into
a 400 at chat time. Probe the boundary values per model family first:

```python
for effort in ["minimal", "low", "medium", "high", "xhigh"]:
    # POST /v1/responses with reasoning: {effort} → expect 200
```

Verified acceptance (2026-11): GPT families accept minimal/low/medium/high/
xhigh; grok accepts low/medium/high/xhigh; gemini-3-flash accepts low/medium/
high; claude-sonnet-5-5 accepts up to max (adaptive). Current declarations:
GPT → six levels; grok → five; gemini-3-flash → four; claude-sonnet-5-5 →
off/low/medium/high/xhigh/max; other claude → four.

## 4. Default Level Semantics (which level applies when the user picks none)

The menu's **"default" item** (i18n `effort.providerDefault`, key
`provider-default`) is built in `dsh-client-ui-model-selection` as
`{effort: undefined}` and is prepended ONLY when no default level is
configured (`reasoning.defaultEffort === void 0`). Selecting it sends NO
reasoning field on the wire (pi-ai: `effort === void 0 → return void 0`) —
it is neither a dsh-side adaptive algorithm nor "thinking off"; the
model/endpoint decides (see layer 1 measurements below). Once a default
level exists (provider `reasoning:` or picker-saved), the separate item
disappears and that level is marked instead.

Three layers, later overrides earlier:

1. **Model/endpoint default** — no `reasoning` field is sent. For
   `openai-responses` the upstream decides (typically ~medium); for
   `anthropic-messages` pi-ai keeps thinking disabled unless an effort is
   active (verified in `streamSimple`).
   Measured on the token-wormhole gateway (2026-11-19, `/v1/responses` usage
   `reasoning_tokens`, two tasks — simple word problem + divisibility proof):
   `deepseek-v4.1-flash` default ≈ **high or above** (113≈118 and 302>249),
   `minimal` = 0 (thinking fully off); `gpt-6.1-sol` default is adaptive
   (auto-throttles on easy tasks). Rule of thumb: **picking nothing does NOT
   mean shallow thinking** — it means the model decides per task.
2. **Profile route default** — add `reasoning: high` on the **provider** entry
   (sibling of `api`/`baseURL`): the level used while a session has picked
   none. CAUTION: it is validated against *every* model's declared levels on
   that route — on a mixed provider (declared + undeclared models) this can
   raise `UNSUPPORTED_REASONING_EFFORT`; set it only on providers where all
   models declare the level (e.g. the Claude-only provider).
3. **Picker-saved default** — choosing an effort in the picker saves it with
   the model as the default for new sessions of that profile (UI state, not
   the patch file). This is the "default" marker visible in the menu.

## 5. Image (vision) input — declare `input:` per model

Symptom: image uploads fine, but sending it to a vision-capable gateway model
(e.g. gpt-6.1-sol) errors "model does not support image". Cause: pi-ai resolves
a model's modalities as `declaredInput(entry.input) ?? base?.input ??
defaultInput(["text"])`. Gateway-custom model ids are NOT in dsh's built-in
base catalog, so an entry without `input:` falls back to **text-only**, and
pi-ai throws `pi-ai model "<id>" does not support image input` when the
request contains an image. It is a missing capability declaration, not a code
bug. Fix: add `input: [text, image]` to the model entry in the patch (the
Settings→Models page's `inputModalities` multi-select writes the same field;
enabling `image` also unlocks `imagePixelBudget`). The model catalog
hot-reloads — no restart. Verify: `dsh --profile web --dump-config` shows
`input: [text, image]` under the model. Apply the same declaration to every
vision-capable gateway model you add (gpt-6.1-terra, gemini, claude, …).

### Bulk fix + audit (a gateway key group usually holds 20+ models)

Declaring `input:` one entry at a time across a 26-model list is slow and
easy to miss. Enumerate → patch → audit:

```bash
P=~/.dsh/profiles/web/cordis.patch.yml
grep -nE '^\s+- id: ' "$P"     # every model id in the patch; keep 12/14-space indent
# insert `input: [text, image]` immediately after each vision model's `name:` line

# audit: any model whose reasoningEfforts block is reached without an input: line
awk '/- id: /{id=$3; has=0; next}
     /input: \[text, image\]/{has=1}
     /reasoningEfforts:/{if(id!="" && !has) print "MISSING: "id; id=""}' "$P"

# compose check + count (absolute dsh path; non-interactive shells may lack PATH)
~/.npm-global/bin/dsh --profile web --dump-config >/tmp/dsh-dump.txt; echo "exit=$?"
grep -c 'input:' /tmp/dsh-dump.txt   # equals the number of vision-declared models
```

The text-only fallback is silent until the first image request, so re-run the
audit after every model-list change (key rotation, new gateway model, UI save).
Two ordering traps: (a) `dsh` is often missing from a detached/agent shell's
PATH — `nohup: failed to run command 'dsh': No such file or directory`, exit
127 — always call `~/.npm-global/bin/dsh` or resolve it via `command -v`;
(b) the Settings→Models page writes this same patch file from its own client
state, so don't hand-edit while that page holds unsaved changes (`never edit
both sides at once` from Phase C).
