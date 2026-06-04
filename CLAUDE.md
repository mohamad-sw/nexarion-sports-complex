# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Requires Python 3.12 and a running [Ollama](https://ollama.com) instance at `http://localhost:11434` with the `gemma3:1b` model pulled.

```bash
pipenv install       # install dependencies
pipenv run python app.py   # run the chatbot
```

## Architecture

Single-file Python project (`app.py`) built on [DSPy](https://dspy.ai).

**LM backend:** DSPy is configured to use a local Ollama model (`ollama/gemma3:1b`). The `api_key` field is a dummy value required by DSPy's interface but unused by Ollama.

**`DoubleChainOfThoughModule`** is a two-stage DSPy pipeline:
1. `cot1` — takes a `question` and produces a `step_by_step_thought` via `ChainOfThought`
2. `cot2` — takes the original `question` + the `thought` and produces a `one_word_answer` via a second `ChainOfThought`

This pattern (chaining CoT modules where each stage's output feeds the next) is the core DSPy idiom used here. New modules should follow the same `dspy.Module` + `forward()` convention.
