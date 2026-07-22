from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
from huggingface_hub import login
import json
import os
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_QW")
model_id = "Qwen/Qwen2.5-32B-Instruct"
use_quantization = False
max_new_tokens = 2000
BATCH_SIZE = 512

print(f"HF_TOKEN found: {HF_TOKEN[:15]}...")
login(HF_TOKEN)

llm = LLM(
    model=model_id,
    tensor_parallel_size=1,
    dtype="bfloat16",
    gpu_memory_utilization=0.90,
    max_model_len=5000,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)

role_instruction = """You are a Medical Risk Explanation Assistant.

Your task is to explain a patient's 30-day hospital readmission risk (provided directly in the input as 'risk_score') by writing a comprehensive clinical synthesis and selecting the most influential factors from the patient's medical record.

STRICT RULES

1. ROLE
- Do NOT make new diagnoses.
- Do NOT recommend treatments, medications, procedures, tests, or follow-up actions.
- Do NOT provide clinical advice.
- Your role is ONLY to explain the provided 'risk_score' using the available medical history.

2. DATA USE & FORBIDDEN FEATURES
- Use ONLY information explicitly present in the input.
- Never invent diagnoses, laboratory values, risk factors, or numerical values.
- SHAP VALUES AS GUIDANCE: You are provided with 'Top Statistical Risk Factors (SHAP values)' as a statistical reference. Treat them as secondary hints, NOT absolute truth. Evaluate them using your clinical knowledge. If a SHAP factor makes strong clinical sense, rely on it. If a SHAP factor lacks strong clinical relevance or justification for this specific patient profile, prioritize the broader medical record (EHR) data instead.
- CRITICAL: Use the EXACT feature names and text string representations from the patient's medical record (EHR) as they appear in the input data. Do not paraphrase or shorten them.
- CRITICAL: Do NOT select, analyze, or mention the following factors under any circumstances: 'length_of_stay', 'insurance', 'admission_type', 'admission_location', or 'discharge_location'.

3. RISK SUMMARY & FACTOR ALIGNMENT
- Both the 'risk_summary' narrative and the selected 'factors' list must strictly align with the provided 'risk_score':
  * HIGH RISK PROFILE: If the provided risk_score is HIGH, the 'risk_summary' must deeply explain the synergistic negative impacts, and the 'factors' list must focus primarily on high-impact risk drivers (effect: "increases_risk").
  * LOW RISK PROFILE: If the provided risk_score is LOW, the 'risk_summary' must deeply explain protective clinical components or stabilized conditions, and the 'factors' list must focus primarily on protective factors or stable clinical markers that keep the risk low (effect: "decreases_risk").
- The 'risk_summary' text must be highly useful for doctors, explicitly stating *why* and *how* these specific values/conditions influence the risk, while remaining compact and avoiding generic filler text.

4. FACTOR SELECTION & MEDICAL CODE FORMATTING
- Select a minimum of 5 and a maximum of 10 factors that best justify the provided 'risk_score'.
- Order factors from the strongest contributor to the weakest contributor.
- EXACT NAMING REQUIREMENT: The "factor" field must match the exact string key from the input. 
- MEDICAL CODES RULE (ICD & CCSR): 
  * If the factor is a specific diagnosis string from the 'icd' array (e.g., "Gastroesophageal reflux disease without esophagitis (K219)") or from the 'ccsr' array (e.g., "Obesity (END009)"), you must use that full string as the "factor" name.
  * For any such ICD or CCSR factor present in the record, set the "value" field strictly to "1" (as a string or number).
- OTHER FEATURES RULE: For all other feature types (laboratory_values, clinical_indicators, demographics), use the exact key string as the "factor" and copy their exact numerical or textual value into the "value" field.

5. OUTPUT FORMAT
- Return ONLY one valid JSON object.
- Do NOT use Markdown.
- Do NOT wrap the JSON inside ```json or ``` blocks.
- The JSON must contain exactly the keys shown below. Do not add extra fields.

JSON schema

{
  "risk_summary": "<200-300 words of detailed, thorough, clinician-oriented cause-and-effect medical explanation of the overall risk score and how the top selected factors drove this specific probability. Must be highly informative for doctors but compact.>",
  "factors": [
    {
      "rank": 1,
      "factor": "<exact feature name string from input EHR data>",
      "value": "<exact numerical/text value from input, or '1' for ICD/CCSR code strings>",
      "effect": "increases_risk | decreases_risk"
    }
  ],
  "limitations": "<empty string if sufficient data; otherwise explain what clinical information is missing to fully explain the risk>"
}
"""

with open('./data/all_patients.json', 'r', encoding='utf-8') as f:
    patient_jsons = json.load(f)

with open('./data/shap_bck_all_patients.json', 'r', encoding='utf-8') as f:
    shap_back_list = json.load(f)

shap_back = {}
for item in shap_back_list:
    sid = item["subject_id"]
    hid = item["hadm_id"]
    shap_back[(sid, hid)] = item["shap_bck_values"]

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=max_new_tokens
)

CONTEXT_TYPES = {
    "row_column": "row_column_context",
    "text": "unstructured_context",
    "long": "long_list_context",
    "json": "json_context",
    "empty": "empty_context",
    "incomplete": "incomplete_context",
}

results = {
    context_name: []
    for context_name in CONTEXT_TYPES
}
patients = patient_jsons["patients"]
total_patients = len(patients)
timing_results = {}

for context_name, context_key in CONTEXT_TYPES.items():
    context_inference_time = 0.0
    num_batches = 0

    print("\nProcessing {context_name.upper()}")

    all_prepared_prompts = []
    for p in patients:
        sid = p["subject_id"]
        hid = p["hadm_id"]
        context = p[context_key]
        if context_name in ["empty", "incomplete"]:
            risk_score = context.get('risk_score', 0.0)
            raw_shap = context.get('shap_bck_values', {})
            context_clean = context.copy()
            context_clean.pop('risk_score', None)
            context_clean.pop('shap_bck_values', None)
            context = context_clean

        else:
            risk_score = p['risk_score']
            raw_shap = shap_back.get((sid, hid), {})
        
        top_10_items = list(raw_shap.items())[:10]
        shap_strings = [f"- {factor}: {value:.4f}" for factor, value in top_10_items]
        shap_context_text = "\n".join(shap_strings)

        if context_name == "row_column":
            context_text = pd.DataFrame(context).to_markdown(index=False)
        elif context_name == "long":
            context_text = "\n".join(context)
        elif isinstance(context, str):
            context_text = context
        else:
            context_text = json.dumps(context, ensure_ascii=False, indent=2)

        prompt = f"""Analyze the following patient record.

        Predicted 30-day readmission risk:
        {risk_score}

        Top Statistical Risk Factors (SHAP values):
        {shap_context_text}

        Patient record:

        {context_text}

        Return only the JSON object."""

        messages = [
            {"role": "system", "content": role_instruction},
            {"role": "user", "content": prompt}
        ]

        templated_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        all_prepared_prompts.append({
            "templated_text": templated_text,
            "meta": (sid, hid),
            "text_length": len(templated_text)
        })

    all_prepared_prompts.sort(key=lambda x: x["text_length"])

    for start in range(0, total_patients, BATCH_SIZE):

        batch_data = all_prepared_prompts[start:start + BATCH_SIZE]
        batch_texts = [item["templated_text"] for item in batch_data]
        batch_meta = [item["meta"] for item in batch_data]

        infer_start = time.perf_counter()

        outputs = llm.generate(
            batch_texts,
            sampling_params
        )
 
        infer_time = time.perf_counter() - infer_start
        context_inference_time += infer_time
        num_batches += 1

        for output, (sid, hid) in zip(outputs, batch_meta):
            response = output.outputs[0].text.strip()
            results[context_name].append({
                "subject_id": sid,
                "hadm_id": hid,
                "context_type": context_name,
                "explanation": response
            })

        print(f"[{context_name}] ")
        print(f"Processed ")
        print(f"{min(start+BATCH_SIZE,total_patients)}/{total_patients}")

    timing_results[context_name] = {
    "total_inference_time_sec": context_inference_time,
    "average_batch_time_sec": context_inference_time / num_batches,
    "average_patient_time_sec": context_inference_time / total_patients,
    "patients": total_patients,
    "batches": num_batches
}

all_results = {
    "model": "Qwen",
    "timings": timing_results,
    "contexts": {}
}

for context_name in CONTEXT_TYPES:
    all_results["contexts"][context_name] = results[context_name]

with open("qwen_results_not_full.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)
print("Done")