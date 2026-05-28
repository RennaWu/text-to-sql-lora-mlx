"""
Gradio demo — side-by-side comparison of base Qwen 2.5-3B (via Ollama)
vs LoRA fine-tuned model (via MLX) on Text-to-SQL generation.

Prerequisites:
  - Run 02_fine_tuning_full.ipynb to produce ./fused_model/
  - Install Ollama and pull qwen2.5:3b for the base model comparison:
        brew install ollama
        ollama pull qwen2.5:3b
  - pip install gradio mlx-lm requests

Run:
  python demo/app.py
"""

import gradio as gr
import requests
from mlx_lm import load, generate

SYSTEM_PROMPT = """You are a SQL expert. Given a database schema and a natural language question, generate the correct SQL query and a brief explanation.

Respond in this exact format:
SQL: <your sql query>
Explanation: <brief explanation>"""

FUSED_DIR = "fused_model"

# Load fine-tuned model at startup
print("Loading fine-tuned model for demo...")
ft_model, ft_tokenizer = load(FUSED_DIR)


def query_base_ollama(prompt, system=SYSTEM_PROMPT):
    """Query the base Qwen 2.5-3B model via local Ollama server."""
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:3b",
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 300},
            },
            timeout=60,
        )
        return resp.json().get("response", "ERROR: No response field")
    except Exception as e:
        return f"Error connecting to Ollama: {e}\nMake sure Ollama is running: `ollama serve`"


def query_finetuned_mlx(prompt, system=SYSTEM_PROMPT):
    """Query the fine-tuned model via MLX."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    p = ft_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return generate(ft_model, ft_tokenizer, prompt=p, max_tokens=300)


def generate_sql(schema, question, model_choice):
    """Dispatch to the chosen model and return its generated SQL."""
    prompt = f"""Database Schema:
{schema}

Question: {question}"""

    if model_choice == "Fine-Tuned":
        return query_finetuned_mlx(prompt)
    return query_base_ollama(prompt)


def build_demo():
    return gr.Interface(
        fn=generate_sql,
        inputs=[
            gr.Textbox(
                label="Database Schema",
                placeholder="CREATE TABLE employees (id INT, name VARCHAR(100), salary DECIMAL);",
                lines=5,
            ),
            gr.Textbox(
                label="Natural Language Question",
                placeholder="What is the average salary?",
                lines=2,
            ),
            gr.Radio(
                choices=["Fine-Tuned", "Base Model"],
                label="Model",
                value="Fine-Tuned",
            ),
        ],
        outputs=gr.Textbox(label="Generated SQL + Explanation", lines=8),
        title="Text-to-SQL Generator",
        description="Generate SQL queries from natural language. Compare base vs fine-tuned model.",
        examples=[
            [
                "CREATE TABLE employees (id INT, name VARCHAR(100), department VARCHAR(50), salary DECIMAL(10,2));\nCREATE TABLE departments (id INT, name VARCHAR(50), budget DECIMAL(12,2));",
                "What is the average salary by department?",
                "Fine-Tuned",
            ],
            [
                "CREATE TABLE orders (id INT, customer_id INT, amount DECIMAL, order_date DATE);\nCREATE TABLE customers (id INT, name VARCHAR(100), city VARCHAR(50));",
                "Find the top 5 customers by total order amount",
                "Fine-Tuned",
            ],
        ],
    )


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=False)
