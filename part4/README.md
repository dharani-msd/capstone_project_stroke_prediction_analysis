**##Part 4**

                        **LLM-Powered Feature Model Prediction Explanation**

**Selected Feature Track: (C) Model Prediction Explanation Pipeline:**

This feature loads the best-performing model (best_model.pkl) from Part 3 and predicts the stroke risk for three hand-crafted patient records. The predicted class and probability are then sent to a Large Language Model (LLM), which returns a structured JSON explanation. Each response is validated against a predefined schema, and a PII guardrail is applied before every LLM API call to ensure no sensitive personal information is included.

**##TASK – 1:**

**##LLM API Connection:**

For the LLM feature, the OpenRouter API was used with the Python requests library. The API key was stored securely in a Google Colab Secret and loaded into the LLM_API_KEY environment variable to avoid hardcoding sensitive information in the notebook.

A reusable function named call_llm() was created to send HTTP POST requests to the OpenRouter API. The function constructs a JSON payload containing the model, messages, temperature, and maximum token limit, sends the request, checks for a successful response, and returns the generated text.

The connection was verified by sending the prompt "Reply with only the word: hello", and the API correctly returned "hello", confirming that the API connection was working successfully.

**##TASK – 2:**

**##Prompt design:**

**System Prompt**

You are a medical prediction explanation assistant.

You will receive:

- Patient feature values
- A predicted class
- A predicted probability

Your task is to explain the prediction and return ONLY valid JSON.

**Return exactly this JSON format:**

{

  "prediction_label": "string",
  
  "confidence_level": "low | medium | high",
  
  "top_reason": "string",
  
  "second_reason": "string",
  
  "next_step": "string"
  
}

Do not include markdown.

Do not include code fences.

Do not include any extra text outside the JSON.

**User Prompt Template:**

Patient Features

Age: {age}

Hypertension: {hypertension}

Heart Disease: {heart_disease}

BMI: {bmi}

Predicted Class:

{prediction}

Predicted Probability:

{probability}

Explain this prediction and return only valid JSON.

**Temperature Varaiations:**

•	The temperature was set to 0 because this task requires consistent and deterministic JSON output. With a temperature of 0, the model produces the same structured response for the same input, making the output easier to validate automatically.

•	A higher temperature, such as 0.7, introduces more randomness, resulting in different wording or explanations while keeping the overall meaning similar.

**Temperature Comparison**

**Input	Output at Temp = 0	Output at Temp = 0.7	Key Difference**

Patient Record 1	Consistent JSON explanation	More varied wording	Temperature 0 produces deterministic output, while temperature 0.7 uses more varied language.

Patient Record 2	Consistent JSON explanation	Slightly different explanation	Higher temperature introduces more variation in the wording.

Patient Record 3	Consistent JSON explanation	More descriptive explanation	Temperature 0.7 provides more diverse phrasing while preserving the same prediction.

**##Task – 3:**

**##Structured output handling:**

 **Model Prediction Explanation Pipeline (Track C)**
 
**Objective**

The best-performing model from Part 3 (best_model.pkl) was loaded using joblib.load(). Three hand-crafted patient records were created as Python dictionaries containing the same feature names used during model training.

**Workflow**

1.	Loaded the saved Random Forest pipeline (best_model.pkl).
2.	Created three sample patient records with different risk profiles.
3.	Passed each record to the trained model to generate:
 
      •	Predicted class (Stroke or No Stroke)
  
      •	Predicted probability using predict_proba()

4.	Sent the patient features, predicted class, and predicted probability to the LLM through the API.
5.	Requested a structured JSON explanation containing:

      •	prediction_label
  
      •	confidence_level
  
      •	top_reason
  
      •	second_reason
  
      •	next_step

6.	Parsed the LLM response using json.loads().
7.	Validated the JSON output using jsonschema.validate().
8.	If the JSON was invalid or failed schema validation, a fallback dictionary with null values was returned instead of causing the program to fail.

**Results**

The model successfully generated predictions for all three sample patient records. The LLM returned structured JSON explanations, and the responses were validated against the predefined schema. This demonstrated how a machine learning model and an LLM can work together: the machine learning model performs the prediction, while the LLM explains the prediction in a structured, human-readable format.

**Note**

The predicted class is determined entirely by the trained machine learning model. The LLM does not make or change the prediction; it only explains the prediction returned by the model.

**Fallback Handling**

The LLM response was parsed using json.loads() and validated using jsonschema.validate(). Two failure scenarios were tested:

  • Invalid JSON: The LLM returned plain text instead of JSON, causing a JSONDecodeError.
  
  • Schema Validation Failure: The LLM returned valid JSON that was missing required fields, causing a ValidationError.
  
In both cases, the program did not crash. Instead, it returned a fallback dictionary with all required fields set to null, ensuring consistent output and graceful error handling.

**##Task – 4:**

**##PII Guardrails:**

To protect user privacy, a PII (Personally Identifiable Information) guardrail was implemented before every LLM API call.

A regular expression (regex) check was used to detect:

  • Email addresses
    
  • Phone numbers
  
If PII is detected, the program does not send the input to the LLM. Instead, it prints:

**Input blocked:PII detected** and skips the API call.

**Two test cases were used to verify the guardrail:**

**Test Input	Expected Result	Outcome**

-Input containing an email address (e.g., john@gmail.com)	LLM call blocked	Passed

-Input containing only patient features (no email or phone number)	LLM call allowed	Passed

This guardrail helps prevent sensitive personal information from being sent to the LLM and demonstrates a basic privacy protection mechanism.

**##Task – 5:**

**##End-to-End Pipeline Demonstration:**

The complete Track C pipeline was executed on three hand-crafted patient records. For each input, the trained machine learning model generated a prediction and probability. Before calling the LLM, a PII guardrail checked the input for email addresses and phone numbers. Since no PII was detected in the three patient records, all requests were allowed to proceed. The LLM returned structured JSON explanations, and each response was successfully validated against the predefined JSON schema.

**Input	LLM Output	Valid JSON	Pass/Block**

**Task – 5:**
**End-to-End Pipeline Demonstration:**

The complete Track C pipeline was executed on three hand-crafted patient records. For each input, the trained machine learning model generated a prediction and probability. Before calling the LLM, a PII guardrail checked the input for email addresses and phone numbers. Since no PII was detected in the three patient records, all requests were allowed to proceed. The LLM returned structured JSON explanations, and each response was successfully validated against the predefined JSON schema.

| Input                                               | LLM Output                | Valid JSON | Pass/Block |
| --------------------------------------------------- | ------------------------- | ---------- | ---------- |
| Patient 1 (Age 65, Hypertension=1, Heart Disease=1) | JSON explanation returned | Pass       | Pass       |
| Patient 2 (Age 30, Healthy profile)                 | JSON explanation returned | Pass       | Pass       |
| Patient 3 (Age 78, Multiple risk factors)           | JSON explanation returned | Pass       | Pass       |

Demonstrate the guardrail blocking a request, include a separate example in the notebook where an input contains an email address and the result is:

| Input                         | LLM Output      | Valid JSON | Pass/Block |
| ----------------------------- | --------------- | ---------- | ---------- |
| Patient with `john@gmail.com` | Not sent to LLM | N/A        | **Block**  |


However, for the required three Track C feature vectors, they normally contain no PII, so all three would show Pass in the guardrail column.
