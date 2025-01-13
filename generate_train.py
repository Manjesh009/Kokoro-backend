import pandas as pd
import yaml
import re
import os

# Extended keyword-to-intent mapping
KEYWORD_TO_INTENT = {
    "heart": "ask_heart_health",
    "pain": "ask_pain",
    "test": "ask_testing",
    "family": "ask_family",
    "hello": "greet",
    "hi": "greet",
    "bye": "goodbye",
    "goodbye": "goodbye",
    "arrhythmia": "ask_arrhythmia",
    "coronary artery disease": "ask_cad",
    "rheumatic heart disease": "ask_rheumatic_heart_disease",
    "endocarditis": "ask_endocarditis",
    "stroke": "ask_stroke",
    "hypertension": "ask_hypertension",
    "high blood pressure": "ask_hypertension",
    "valvular heart disease": "ask_valvular_heart_disease",
    "aortic aneurysm": "ask_aortic_aneurysm",
    "cardiomyopathy": "ask_cardiomyopathy",
    "heart attack": "ask_heart_attack",
    "myocardial infarction": "ask_heart_attack",
    "pericarditis": "ask_pericarditis",
    "peripheral artery disease": "ask_pad",
    "pad": "ask_pad",
    "congenital heart disease": "ask_congenital_heart_disease",
}

def clean_intent_name(intent_name):
    """Sanitize the intent name by replacing invalid characters."""
    # Replace non-alphanumeric characters with underscores
    intent_name = re.sub(r"[^a-zA-Z0-9_]+", "_", intent_name)
    # Remove leading or trailing underscores
    intent_name = intent_name.strip("_")
    return intent_name

def get_intent_from_question(question):
    """Determine the intent based on the keywords in the question."""
    for keyword, intent in KEYWORD_TO_INTENT.items():
        if keyword.lower() in question.lower():
            return intent
    return "ask_general"  # Default intent for unmatched questions

def process_data(dataframe):
    """Process the Excel data to generate intents and responses."""
    intents = {}
    responses = {}

    for _, row in dataframe.iterrows():
        if pd.isna(row["Question"]) or pd.isna(row["Answer"]):
            continue

        question = row["Question"].strip()
        answer = row["Answer"].strip()

        # Determine intent based on keywords
        intent = get_intent_from_question(question)

        # Group questions under the same intent
        if intent not in intents:
            intents[intent] = []
        intents[intent].append(question)

        # Add responses
        if intent not in responses:
            responses[intent] = []
        responses[intent].append(answer)

    return intents, responses

def write_nlu_yaml(intents, filename="data/nlu.yml"):
    """Write the NLU data to a YAML file."""
    nlu_data = {"version": "3.1", "nlu": []}
    for intent, examples in intents.items():
        nlu_data["nlu"].append({
            "intent": intent,
            "examples": "\n".join([f"- {example}" for example in examples])
        })

    with open(filename, "w", encoding="utf-8") as file:
        yaml.dump(nlu_data, file, allow_unicode=True)

def write_domain_yaml(responses, filename="domain.yml"):
    """Write the domain data to a YAML file."""
    domain_data = {
        "version": "3.1",
        "intents": list(responses.keys()),
        "responses": {}
    }
    for intent, texts in responses.items():
        domain_data["responses"][f"utter_{intent}"] = [{"text": text} for text in texts]

    with open(filename, "w", encoding="utf-8") as file:
        yaml.dump(domain_data, file, allow_unicode=True)

def write_stories_yaml(intents, filename="data/stories.yml"):
    """Write the stories to a YAML file."""
    stories_data = {"version": "3.1", "stories": []}
    for intent in intents.keys():
        stories_data["stories"].append({
            "story": f"Story for {intent}",
            "steps": [
                {"intent": intent},
                {"action": f"utter_{intent}"}
            ]
        })

    with open(filename, "w", encoding="utf-8") as file:
        yaml.dump(stories_data, file, allow_unicode=True)

def write_rules_yaml(intents, filename="data/rules.yml"):
    """Write the rules to a YAML file."""
    rules_data = {"version": "3.1", "rules": []}
    for intent in intents.keys():
        rules_data["rules"].append({
            "rule": f"Rule for {intent}",
            "steps": [
                {"intent": intent},
                {"action": f"utter_{intent}"}
            ]
        })

    with open(filename, "w", encoding="utf-8") as file:
        yaml.dump(rules_data, file, allow_unicode=True)

def main():
    # Load data from the Excel file
    excel_file = "kokoro_chatbot_testing.xlsx"  # Replace with your file path
    if not os.path.exists(excel_file):
        print(f"❌ Excel file '{excel_file}' not found!")
        return

    df = pd.read_excel(excel_file)
    intents, responses = process_data(df)

    # Generate training files
    write_nlu_yaml(intents)
    print("✅ NLU data written to 'data/nlu.yml'")
    write_domain_yaml(responses)
    print("✅ Domain data written to 'domain.yml'")
    write_stories_yaml(intents)
    print("✅ Stories data written to 'data/stories.yml'")
    write_rules_yaml(intents)
    print("✅ Rules data written to 'data/rules.yml'")

if __name__ == "__main__":
    main()
