from transformers import BitsAndBytesConfig, pipeline
import torch
from huggingface_hub import login
import json
import os
from clearml import Task
import time
import requests
import pandas as pd

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
task = Task.init(
    project_name="pershin-medailab/LLM_verification_risk_profiles",
    task_name="MedGemma Inference with SHAP",
    #output_uri="s3://api.blackhole2.ai.innopolis.university:443/pershin-medailab"
    output_uri=None,
    auto_connect_arg_parser=False,
    auto_connect_frameworks=False   
)
HF_TOKEN = None
RECEIVER_URL = "https://elective-zipping-drum.ngrok-free.dev"
model_variant = "medgemma-27b-it"
model_id = f"google/{model_variant}"
use_quantization = True
max_new_tokens = 2000
BATCH_SIZE = 32

config_params = {
    "model": model_variant,
    "quantization": use_quantization,
    "max_new_tokens": max_new_tokens,
    "batch_size": BATCH_SIZE,
    "HF_TOKEN": ""
}
task.connect(config_params) 

if config_params.get("HF_TOKEN"):
    HF_TOKEN = config_params["HF_TOKEN"]

if not HF_TOKEN:
    HF_TOKEN = os.environ.get("HF_TOKEN")


if not HF_TOKEN:
    try:
        HF_TOKEN = task.get_parameter("General/HF_TOKEN") or task.get_parameter("Args/HF_TOKEN")
    except:
        pass

if not HF_TOKEN:
    raise ValueError("No HF_TOKEN")
print(f"HF_TOKEN found: {HF_TOKEN[:15]}...")
login(HF_TOKEN)
print(os.getcwd())

num_gpus = torch.cuda.device_count()
model_kwargs = dict(
    dtype=torch.bfloat16,
    device_map="auto"
)

if num_gpus > 1:
    model_kwargs["device_map"] = "auto"
    print("Multi-GPU")
    BATCH_SIZE = 52
else:
    model_kwargs["device_map"] = "auto" 
    print("Single GPU")

if use_quantization:
    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

if "text" in model_variant:
    pipe = pipeline("text-generation", model=model_id, model_kwargs=model_kwargs)
else:
    pipe = pipeline("image-text-to-text", model=model_id, model_kwargs=model_kwargs)


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
- SHAP VALUES AS GUIDANCE: You are provided with 'Top 10 Statistical Risk Factors (SHAP values)' as a statistical reference. Treat them as secondary hints, NOT absolute truth. Evaluate them using your clinical knowledge. If a SHAP factor makes strong clinical sense, rely on it. If a SHAP factor lacks strong clinical relevance or justification for this specific patient profile, prioritize the broader medical record (EHR) data instead.
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

with open('main_generations/data/all_patients.json', 'r', encoding='utf-8') as f:
    patient_jsons = json.load(f)

with open('main_generations/data/shap_bck_all_patients.json', 'r', encoding='utf-8') as f:
    shap_back_list = json.load(f)

shap_back = {}
for item in shap_back_list:
    sid = item["subject_id"]
    hid = item["hadm_id"]
    shap_back[(sid, hid)] = item["shap_bck_values"]


CONTEXT_TYPES = {
    "long": "long_list_context",
    "row_column": "row_column_context",
    "json": "json_context",
    "text": "unstructured_context",
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
    task.get_logger().report_text(
        f"\nProcessing {context_name.upper()}"
    )

    all_prepared_prompts = []
    
    for p in patients:
        sid = p["subject_id"]
        hid = p["hadm_id"]
        context = p[context_key]
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
            
        risk_score = p['risk_score']

        prompt = f"""Analyze the following patient record.

        Predicted 30-day readmission risk:
        {risk_score}

        Top 10 Statistical Risk Factors (SHAP values):
        {shap_context_text}

        Patient record:

        {context_text}

        Return only the JSON object."""

        messages = [
            {"role": "system", "content": [{"type": "text", "text": role_instruction}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        
        all_prepared_prompts.append({
            "messages": messages,
            "meta": (sid, hid),
            "text_length": len(prompt)
        })

    all_prepared_prompts.sort(key=lambda x: x["text_length"])

    for start in range(0, total_patients, BATCH_SIZE):
        batch_data = all_prepared_prompts[start:start + BATCH_SIZE]
        batch_messages = [item["messages"] for item in batch_data]
        batch_meta = [item["meta"] for item in batch_data]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        infer_start = time.perf_counter()
        with torch.inference_mode():
            outputs = pipe(
                batch_messages,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pipe.tokenizer.eos_token_id,
                batch_size=len(batch_messages)
            )

        infer_time = time.perf_counter() - infer_start
        context_inference_time += infer_time
        num_batches += 1

        for output, (sid, hid) in zip(outputs, batch_meta):
            response = output[0]["generated_text"][-1]["content"]
            results[context_name].append(
                {
                    "subject_id": sid,
                    "hadm_id": hid,
                    "context_type": context_name,
                    "explanation": response
                }
            )

        task.get_logger().report_text(
            f"[{context_name}] "
            f"Processed "
            f"{min(start+BATCH_SIZE,total_patients)}/{total_patients}"
        )

    timing_results[context_name] = {
    "total_inference_time_sec": context_inference_time,
    "average_batch_time_sec": context_inference_time / num_batches,
    "average_patient_time_sec": context_inference_time / total_patients,
    "patients": total_patients,
    "batches": num_batches
}

# for context_name in CONTEXT_TYPES:

#     filename = f"inference_results_{context_name}.json"
#     print(results[context_name])

    # with open(filename, "w", encoding="utf-8") as f:
    #     json.dump(
    #         results[context_name],
    #         f,
    #         ensure_ascii=False,
    #         indent=2
    #     )

    # task.upload_artifact(
    #     name=f"medgemma_{context_name}",
    #     artifact_object=filename
    # )

def send_results_to_notebook(data, description=""):
    try:
        response = requests.post(
            RECEIVER_URL,
            json=data,
            timeout=30
        )
        if response.status_code == 200:
            print(f"Sent to notebook: {description}")
            task.get_logger().report_text(f"Sent to notebook: {description}")
            return True
        else:
            print(f"Failed ({response.status_code}): {response.text}")
            return False
    except Exception as e:
        print(f"Error sending {description}: {e}")
        task.get_logger().report_text(f"Send failed: {str(e)}")
        return False

all_results = {
    "task_id": task.id,
    "task_name": task.name,
    "timestamp": time.time(),
    "timings": timing_results,
    "contexts": {}
}
for context_name in CONTEXT_TYPES:
    all_results["contexts"][context_name] = results[context_name]

send_results_to_notebook(all_results, "all_results_combined")

all_results = {}
for context_name in CONTEXT_TYPES:
    all_results[context_name] = results[context_name]

all_results['timings'] = timing_results
task.upload_artifact(
    name="all_inference_results",
    artifact_object=all_results,
    metadata={"type": "combined_results"}
)

print("Done")
time.sleep(10)
task.close()