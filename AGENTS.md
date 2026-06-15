# AGENTS.md

Behavioral guidelines for AI coding agents working on **VibeVoice** — a research-oriented voice AI framework (TTS / ASR / streaming). Merge with project-specific instructions as needed.

**Tradeoff:** these guidelines bias toward caution over speed. For trivial tasks, use judgment.

---

## 0. Project Context

VibeVoice is a **research codebase**, not a commercial product. Core principles from [CONTRIBUTING.md](CONTRIBUTING.md) are non-negotiable:

- **Code minimalism, high readability, functional purity.**
- **No over-engineering** — no premature abstractions, no speculative "flexibility."
- **Style-only PRs are rejected.** Don't refactor adjacent code while you're there.
- **English only** — code, comments, commits, docs. Non-English contributions are rejected.
- **Line-by-line human review.** Every line must justify its existence.

Key paths:
- `vibevoice/modular/` — model architectures (`modeling_vibevoice*.py`, tokenizers, diffusion head)
- `vibevoice/processor/` — pre/post-processing (`vibevoice_asr_processor.py`, audio utils)
- `vibevoice/schedule/` — diffusion timestep / DPM-Solver
- `vllm_plugin/` — vLLM integration (`model.py`, `inputs.py`)
- `finetuning-asr/` — LoRA fine-tuning
- `demo/` — Gradio demos + Colab notebooks
- `api.py` — FastAPI ASR server
- `docs/` — model guides (`vibevoice-asr.md`, `vibevoice-tts.md`, …)

Use `rtk` (per the global hook) for shell commands to keep context lean.

---

## 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:
- State assumptions explicitly; ask if uncertain.
- Present multiple interpretations rather than picking silently.
- Suggest simpler approaches; push back when warranted.
- If unclear, stop, name the confusion, and ask.

For VibeVoice specifically:
- Distinguish **research code** (tolerates single-use helpers, in-line magic) from **API/demo code** (must be clean, parameterized, error-handled).
- Check whether a behavior is a model-spec invariant (don't change) vs. an implementation detail (fair game).

---

## 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No unrequested "flexibility" or "configurability."
- No error handling for impossible scenarios.
- If 200 lines could be 50, rewrite it.

**Self-check:** "Would a senior VibeVoice maintainer say this is overcomplicated?" If yes, simplify. The project explicitly rejects "unnecessary encapsulation, excessive abstraction, or complex architectural refactoring."

---

## 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style (variable names, error messages, log format).
- Mention unrelated dead code rather than deleting it.

When changes create orphans:
- Remove imports/variables/functions your changes made unused.
- Don't remove pre-existing dead code unless asked.

**Test:** "Every changed line should trace directly to the user's request." If a line doesn't, cut it or justify it in the commit message.

---

## 4. Goal-Driven Execution

Define success criteria. Loop until verified.

Transform vague asks into verifiable ones:
- "Add validation" → "Write tests for invalid inputs, then make them pass."
- "Fix the bug" → "Write a test that reproduces it, then make it pass."
- "Refactor X" → "Ensure tests pass before and after."

For multi-step tasks, the format is:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

For VibeVoice, "verified" usually means:
- Module imports cleanly: `python -c "import vibevoice"`
- Unit tests pass: `pytest vibevoice/processor/tests/`
- A representative inference call runs end-to-end on a small sample.
- For ASR API: `curl` `/transcribe` with a short audio file and inspect the response.

**Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.**

---

## 5. LLM-Generated Code Caveat

VibeVoice maintainers explicitly flag LLM-generated PRs as risky. When you write code here:

- Avoid redundant logic, defensive copies, and "just in case" branches.
- Don't import utilities you don't use.
- Don't add docstrings, type hints, or comments that paraphrase the code.
- Run the code (or trace it line by line) before claiming it works.

---

## 6. Success Indicator

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, fewer rejections on style, and clarifying questions come before implementation rather than after mistakes.
