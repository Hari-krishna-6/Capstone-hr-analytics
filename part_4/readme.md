# LLM-Powered Feature: Model Prediction Explanation Pipeline (Part 4)

This section details the implementation of **Track C: Model Prediction Explanation Pipeline**. This feature integrates the serialized machine learning pipeline asset generated in Part 3 with a generative Large Language Model layer to provide structured, validated, and secure natural language explanations of algorithmic flight risk scores.

---

## 🎯 Track Selection & Design Decisions

* **Selected Track:** Track C — Model Prediction Explanation Pipeline
* **Temperature Configuration:** `temperature=0.0`
* **Rationale for Temperature Choice:** Setting the temperature to `0.0` forces the LLM to use deterministic decoding. Instead of sampling from a broader distribution of tokens, the model always selects the token with the highest mathematical probability. This eliminates creative variance and ensures that the output strictly adheres to the requested JSON syntax, offering reproducible explanations across automated system runs.

---

## 📝 Verbatim Prompt Specifications

### 1. System Prompt
```text
You are a workforce analytics explanation system. Given specific employee feature values, their predicted attrition classification status (0=Stay, 1=Leave), and the algorithmic probability of flight risk, output a valid JSON object explaining the model's prediction. 

Your output must strictly follow this JSON schema structure without markdown wrappers or code blocks:
{
  "prediction_label": "Clear textual translation of prediction class",
  "confidence_level": "low or medium or high based on closeness to classification margins",
  "top_reason": "Primary driving feature rationale linked to operational attrition context",
  "second_reason": "Secondary supporting feature rationale from input data",
  "next_step": "Strategic, actionable retention management recommendation"
}
Ensure all keys are populated using scalar string text values.
ecurity Guardrail: PII Detection Results
Before any payload is assembled or dispatched to the external LLM API endpoint, the raw user text is processed through a regular expression scanner to identify and block Personally Identifiable Information (PII).

PII Guardrail Regular Expression Patterns:

Email Scanner: r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'

Phone Sequence Scanner: r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
Temperature Variability Analysis (A/B Testing)
Each handcrafted feature-vector record was run through the explanation pipeline at both temperature=0.0 and temperature=0.7 to evaluate structural consistency.

Determinism Rationale
At temperature=0.0, the model acts as a deterministic greedy search engine, picking the single highest-probability next token, making it perfect for schema-bound parsing. Increasing the temperature to 0.7 flattens the probability distribution, allowing less likely tokens to be sampled. This introduces stylistic vocabulary shifts that risk adding conversational text or markdown code blocks, which can cause downstream json.loads() functions to fail.

End-to-End Pipeline Demonstration & Schema Validation
The production pipeline loaded best_model.pkl using joblib, evaluated three hand-crafted feature vectors via .predict() and .predict_proba(), passed the metadata securely through the PII filter, and validated the JSON responses against the structural schema constraints using jsonschema.validate().