from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.vector_store import search_vector_store


# ------------------------------------------------
# Model config
# ------------------------------------------------
model_name = "microsoft/phi-3-mini-4k-instruct"

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = None
model = None


# ------------------------------------------------
# Load model once
# ------------------------------------------------
def load_llm():

    global tokenizer, model

    if tokenizer is None or model is None:

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)

    return tokenizer, model


# ------------------------------------------------
# Clean answer
# ------------------------------------------------
def clean_answer(text):

    # remove any leftover tokens
    text = text.strip()

    # stop after first 3 sentences
    sentences = text.split(".")
    text = ".".join(sentences[:3]).strip()

    if not text.endswith("."):
        text += "."

    return text


# ------------------------------------------------
# Answer generation
# ------------------------------------------------
def answer_question(index, chunks, question):

    tokenizer, model = load_llm()

    # Retrieve most relevant chunk
    context = search_vector_store(index, chunks, question)

    # keep context reasonably small
    context = context[:450]


    prompt = f"""
<|system|>
You are helping a student understand their study notes.

Instructions:
- Read the context carefully.
- Answer based mainly on the study notes.
- Explain clearly in simple textbook language.
- Do not invent unrelated information.
- Write 1–3 sentences.

<|user|>
Study Notes:
{context}

Question:
{question}

Answer clearly using the study notes.

<|assistant|>
"""


    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    input_length = inputs["input_ids"].shape[1]


    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )


    # decode only generated tokens
    generated_tokens = outputs[0][input_length:]

    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    answer = clean_answer(answer)

    return answer