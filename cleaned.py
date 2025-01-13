import yaml
import re

def clean_yaml_keys(file_path):
    try:
        # Load the YAML file
        with open(file_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        
        # Function to clean keys recursively
        def clean_keys(data):
            if isinstance(data, dict):
                return {re.sub(r'[^\w]', '_', key): clean_keys(value) for key, value in data.items()}
            elif isinstance(data, list):
                return [clean_keys(item) for item in data]
            else:
                return data
        
        # Clean the keys in the loaded YAML data
        cleaned_data = clean_keys(yaml_data)
        
        # Write the cleaned YAML back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(cleaned_data, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ Cleaned and updated keys in {file_path}")
    except yaml.YAMLError as e:
        print(f"❌ Error processing {file_path}: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

# Path to your domain.yml file
file_path = 'domain.yml'

# Run the cleaning script
clean_yaml_keys(file_path)
