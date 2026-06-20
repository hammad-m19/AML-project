"""Hugging Face Transformers utilities for QA, text generation, and NER."""

import json
import urllib.request
import urllib.error
from functools import lru_cache

import tf_keras  # noqa: F401 — required for Transformers + Keras 3 compatibility
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)


DEVICE = 0 if torch.cuda.is_available() else -1

# Hugging Face Inference API (free, no key required for public models)
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"


@lru_cache(maxsize=1)
def _text_gen_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        framework="pt",
    )


@lru_cache(maxsize=1)
def _ner_pipeline():
    return pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",
        device=DEVICE,
        framework="pt",
    )


def _query_hf_api(prompt, max_tokens=200):
    """Query Hugging Face free Inference API."""
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        HF_API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    if isinstance(result, list) and len(result) > 0:
        return result[0].get("generated_text", "").strip()
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(result["error"])
    return str(result)


def _fallback_local_qa(question):
    """Fallback: use local distilgpt2 if API is unavailable.
    Uses manual token loop because pipeline/model.generate crashes on this OS/env.
    """
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    with torch.no_grad():
        for _ in range(30):  # limit length to avoid slow generation
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Greedy decoding
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long)], dim=-1)
            
    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    if "Answer:" in text:
        ans = text.split("Answer:")[-1].strip().split("\n")[0].strip()
        return ans if ans else "I don't know."
    return text[len(prompt):].strip().split("\n")[0].strip()


def answer_question(question, context=""):
    """Run generative QA — uses HF Inference API, falls back to local model."""

    # Build the instruction prompt
    if context.strip():
        prompt = (
            f"[INST] Answer the following question based on the context provided. "
            f"Give a clear, detailed answer.\n\n"
            f"Context: {context.strip()}\n\n"
            f"Question: {question.strip()} [/INST]"
        )
    else:
        prompt = (
            f"[INST] Answer the following question clearly and in detail. "
            f"Provide accurate, factual information.\n\n"
            f"Question: {question.strip()} [/INST]"
        )

    try:
        answer_text = _query_hf_api(prompt)

        # Clean up: stop at certain markers
        for stop in ["\n\n\n", "[INST]", "</s>"]:
            if stop in answer_text:
                answer_text = answer_text[:answer_text.index(stop)].strip()

        if not answer_text:
            raise ValueError("Empty response")

        return {
            "answer": answer_text,
            "model": "Mistral-7B-Instruct",
            "mode": "contextual" if context.strip() else "open-ended",
        }
    except Exception:
        # Fallback to local model
        fallback_answer = _fallback_local_qa(question)
        return {
            "answer": fallback_answer if fallback_answer else "Could not generate an answer.",
            "model": "distilgpt2 (local fallback)",
            "mode": "contextual" if context.strip() else "open-ended",
        }


def generate_text(prompt, max_length=120, num_return_sequences=1):
    """Generate text continuation from a prompt."""
    gen = _text_gen_pipeline()
    outputs = gen(
        prompt,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        pad_token_id=gen.tokenizer.eos_token_id,
    )
    return {
        "prompt": prompt,
        "generated_texts": [o["generated_text"] for o in outputs],
    }


def extract_entities(text):
    """Run Named Entity Recognition on input text."""
    ner = _ner_pipeline()
    entities = ner(text)
    formatted = []
    for ent in entities:
        formatted.append(
            {
                "entity": ent.get("entity_group", ent.get("entity", "")),
                "word": ent["word"],
                "score": round(float(ent["score"]), 4),
                "start": int(ent["start"]),
                "end": int(ent["end"]),
            }
        )
    return {"text": text, "entities": formatted, "entity_count": len(formatted)}
