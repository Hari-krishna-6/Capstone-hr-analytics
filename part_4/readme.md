# Part 4 – LLM Powered Feature (Track C: Model Prediction Explanation Pipeline)

## Objective

For this part of the project, I selected **Track C – Model Prediction Explanation Pipeline**. The objective is to combine the machine learning model developed in Part 3 with a Large Language Model (LLM) to generate structured explanations for model predictions.

The pipeline loads the trained model, predicts employee attrition for selected records, obtains prediction probabilities, and sends this information to an LLM. The LLM returns a structured JSON explanation, which is validated before being accepted.

---

# Track Selected

**Track C – Model Prediction Explanation Pipeline**

The complete workflow is:

1. Load the trained model (`best_model.pkl`).
2. Load and preprocess the cleaned dataset.
3. Select three employee records.
4. Predict the class using `.predict()`.
5. Predict the probability using `.predict_proba()`.
6. Send the feature values, predicted class and probability to the LLM.
7. Validate the returned JSON using a predefined schema.
8. Display the explanation.

---

# API Configuration

The API key is securely stored as an environment variable.

```python
API_KEY = os.environ.get("LLM_API_KEY")
```

The notebook does not contain any hardcoded API keys.

A reusable function named **call_llm()** was created using the Python **requests** library. This function:

* creates the JSON request body,
* sends an HTTP POST request,
* checks the response status,
* returns the generated response.

To verify the connection, the following test prompt was executed.

**Prompt**

```text
Reply with only the word: hello
```

**Response**

```text
hello
```

This confirmed that the API connection was working correctly.

---

# Prompt Design

## System Prompt

The system prompt defines the model as an HR Analytics prediction explanation assistant.

The LLM receives:

* Employee feature values
* Predicted class
* Prediction probability

The model is instructed to return **only valid JSON** containing the following fields:

* prediction_label
* confidence_level
* top_reason
* second_reason
* next_step

No markdown or additional text is allowed.

---

## User Prompt Template

```text
Employee Features

{features}

Predicted Class

{prediction}

Prediction Probability

{probability}

Generate only valid JSON.
```

---

# Why Temperature = 0?

Temperature controls the randomness of the generated response.

For this task, structured JSON output is required. Therefore, **temperature = 0** was chosen because it produces deterministic and consistent responses.

A second experiment was performed using **temperature = 0.7** to observe how the explanations changed.

---

# Temperature Comparison

| Record   | Temperature = 0                                                                  | Temperature = 0.7                                                                            | Key Difference                                             |
| -------- | -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Record 1 | Medium confidence explanation focused on overtime and relationship satisfaction. | Medium confidence explanation focused on overtime, work-life balance and previous companies. | Prediction unchanged, wording became more detailed.        |
| Record 2 | Employee predicted to stay with emphasis on salary hike and company tenure.      | Employee predicted to stay with similar reasoning but different wording.                     | Prediction unchanged, explanation wording varied.          |
| Record 3 | Low confidence explanation based on monthly income and environment satisfaction. | Low confidence explanation including job level, overtime and tenure.                         | Prediction unchanged, explanation became more descriptive. |

Temperature **0** produced consistent structured responses, while **0.7** generated more varied explanations for the same prediction.

---

# JSON Schema Validation

A JSON schema was created containing the following required fields:

* prediction_label
* confidence_level
* top_reason
* second_reason
* next_step

After every API call:

1. The response was cleaned using `strip()`.
2. The response was parsed using `json.loads()`.
3. The parsed object was validated using `jsonschema.validate()`.

If validation failed, a fallback dictionary containing null values was returned.

---

# PII Guardrail

Before every API request, the input was checked for Personally Identifiable Information (PII).

The guardrail detects:

* Email addresses
* Phone numbers

### Guardrail Test

| Input                                           | Result  |
| ----------------------------------------------- | ------- |
| [employee@gmail.com](mailto:employee@gmail.com) | Blocked |
| Age is 30 and MonthlyIncome is 5000             | Allowed |

The guardrail successfully prevented sensitive information from being sent to the LLM.

---

# End-to-End Demonstration

Three employee records from the processed dataset were passed through the complete pipeline.

For each record:

* the trained Random Forest model generated a prediction,
* the prediction probability was calculated,
* the feature values, prediction and probability were sent to the LLM,
* the returned JSON was validated successfully.

### Record 1

| Item                   | Value |
| ---------------------- | ----- |
| Predicted Class        | 1     |
| Prediction Probability | 0.630 |
| Validation             | PASS  |

The LLM identified overtime and poor relationship satisfaction as the major reasons for employee attrition and recommended reducing overtime and improving employee engagement.

---

### Record 2

| Item                   | Value |
| ---------------------- | ----- |
| Predicted Class        | 0     |
| Prediction Probability | 0.035 |
| Validation             | PASS  |

The explanation highlighted the employee's long company tenure and high salary hike as indicators of low attrition risk and recommended continued employee engagement.

---

### Record 3

| Item                   | Value |
| ---------------------- | ----- |
| Predicted Class        | 0     |
| Prediction Probability | 0.475 |
| Validation             | PASS  |

The explanation suggested that although the employee had relatively low monthly income, high environment satisfaction reduced the likelihood of attrition. The recommendation was to monitor employee engagement and career growth.

---

# Validation Summary

| Record | Predicted Class | Probability | Validation |
| ------ | --------------: | ----------: | ---------- |
| 1      |               1 |       0.630 | PASS       |
| 2      |               0 |       0.035 | PASS       |
| 3      |               0 |       0.475 | PASS       |

All three responses were successfully parsed and validated against the predefined JSON schema.

---

# Conclusion

In this part of the project, a complete LLM-powered prediction explanation pipeline was successfully implemented.

The trained machine learning model generated employee attrition predictions using `.predict()` and `.predict_proba()`. These predictions, together with the employee feature values, were passed to the LLM to generate structured explanations.

The API key was securely managed through environment variables, the responses were validated using a JSON schema, and a PII guardrail prevented sensitive information from being transmitted.

Finally, three employee records were successfully processed at two different temperature settings. All responses passed schema validation, demonstrating that the pipeline produces reliable, structured and explainable outputs suitable for HR Analytics applications.