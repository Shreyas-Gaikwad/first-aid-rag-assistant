def build_prompt(query, chunks, max_chars=1200):
    context = []
    total = 0

    for c in chunks:
        text = c["text"][:500].strip() + "\n"
        if total + len(text) > max_chars:
            break
        context.append(text)
        total += len(text)

    return f"""### SYSTEM
You are a first-aid reference assistant.

Rules:
- Use ONLY the context.
- Write numbered steps.
- Use simple language.
- Do NOT repeat instructions.
- Do NOT include sources.
- Stop when finished.
- If the person becomes unconscious, advise calling emergency services.

### CONTEXT
{''.join(context)}

### QUESTION
{query}

### ANSWER
"""