# 🧠 AI Interview Bot — README

This project contains two core scripts that use NVIDIA's `AceInstruct-1.5B` language model to generate and continue interview conversations. It uses HuggingFace's `transformers` library and runs on GPU (`device_map="cuda"`).

---

## 📁 File: `question_generation.py`

### 🎯 Purpose
Generates **interview questions** based on a resume or input text using skills and projects mentioned.

### ⚙️ How It Works

1. **Model Setup**
   ```python
   model_name = "nvidia/AceInstruct-1.5B"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="float32", device_map="cuda")
   ```

2. **Prompt Template**
   ```
   Generate questions based on the below resume using Skills and Projects. Only return the questions, with no other data...
   ```

3. **Tokenization & Inference**
   - Applies chat template.
   - Sends input to GPU.
   - Generates up to 4096 tokens.

4. **Post-processing**
   - Removes bullet/number prefixes.
   - Filters out empty lines.
   - Adds an intro question:
     ```
     Hello! I'm excited to start our conversation. How would you like to introduce yourself?
     ```

5. **Returns**
   - A list of clean interview questions.

### ✅ Function Signature
```python
def generate_questions(prompt_text: str) -> list
```

---

## 📁 File: `next_response.py`

### 🎯 Purpose
Handles **follow-up interviewer dialogue** based on past conversation.

### 🌐 Flask Endpoint
```python
@next_response.route('/generate', methods=['POST'])
```

### 🧩 Input
```json
{
  "conversational_history": "..."
}
```

### ⚙️ How It Works

1. **Model Setup**
   - Loads the same `"nvidia/AceInstruct-1.5B"` model.

2. **Prompt Template**
   ```
   Continue the conversation based on the conversation given below, your role is "INTERVIEWER"...
   ```

3. **Tokenization & Inference**
   - Applies chat template.
   - Sends input to GPU.
   - Generates next response.

4. **Returns**
   - A plain text **next line** as the interviewer's response.

---

## 🔄 Workflow Summary

| Step | File | Action |
|------|------|--------|
| 🧾 Resume Input | `question_generation.py` | Generates initial questions |
| 🗣️ Conversation Continuation | `next_response.py` | Generates next line for interviewer |

---

## 🧪 Example Usage

### Generate Questions
```python
questions = generate_questions(resume_text)
```

### Generate Next Interviewer Line (Flask API)
```http
POST /generate
Content-Type: application/json

{
  "conversational_history": "Candidate: I am a software engineer with experience in NLP..."
}
```

---

## 🚀 Dependencies
- `transformers`
- `torch`
- `flask`

---

Let me know if you want a `requirements.txt`, Dockerfile, or front-end integration with this!
