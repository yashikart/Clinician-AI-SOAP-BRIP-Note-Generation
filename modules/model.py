"""
LLM Provider Module - Supports multiple LLM providers

This module provides functionality to query different LLM providers for generating SOAP/BIRP notes.
Supports: OpenAI API, Local models (transformers), AWS Bedrock (optional)

Attributes:
    project_root (str): The root directory of the project.
    config_file_path (str): The path to the config.json file.

Functions:
    query_llm(prompt):
        Query the configured LLM provider with the provided prompt.
"""

import json
import os
import torch
from modules.utils import load_config_data

# Try to import optional dependencies
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Get the path to the config.json file
project_root = os.path.dirname(os.path.dirname(__file__))  # Navigate 2 levels up from the script directory
config_file_path = os.path.join(project_root, 'config.json')


def query_openai(prompt):
    """
    Query OpenAI API with the provided prompt.

    Args:
        prompt (str): The prompt to query OpenAI.

    Returns:
        dict: Response in the same format as Bedrock for compatibility.
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI library not installed. Install with: pip install openai")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenAI API key not found. Set it as an environment variable:\n"
            "  Windows PowerShell: $env:OPENAI_API_KEY = 'your-api-key'\n"
            "  Linux/Mac: export OPENAI_API_KEY='your-api-key'\n\n"
            "Get your API key from: https://platform.openai.com/api-keys"
        )
    
    client = openai.OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Can be changed to gpt-4 if available
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.7
        )
        
        # Return in same format as Bedrock for compatibility
        return {
            "content": [{
                "text": response.choices[0].message.content
            }]
        }
    except Exception as e:
        raise RuntimeError(f"Error calling OpenAI API: {str(e)}") from e


def query_local_model(prompt, model_name=None):
    """
    Query a local model using transformers.
    Uses a smaller, faster model suitable for medical note generation.

    Args:
        prompt (str): The prompt to query the model.
        model_name (str): Hugging Face model name. If None, uses a default model.

    Returns:
        dict: Response in the same format as Bedrock for compatibility.
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers library not installed. Install with: pip install transformers")
    
    # Use a smaller, faster model for local inference
    if model_name is None:
        # Using a smaller model that's good for instruction following
        model_name = "gpt2"  # Small and fast, but can be changed to better models
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    try:
        print(f"Loading local model: {model_name} (this may take a moment on first run)...")
        # Use a text generation pipeline
        generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            device=device if device != "cpu" else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Generate response
        result = generator(
            prompt,
            max_new_tokens=800,
            temperature=0.7,
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id if generator.tokenizer.eos_token_id else generator.tokenizer.pad_token_id
        )
        
        # Extract generated text
        generated_text = result[0]['generated_text']
        # Remove the original prompt from the response
        if len(generated_text) > len(prompt):
            response_text = generated_text[len(prompt):].strip()
        else:
            response_text = generated_text.strip()
        
        return {
            "content": [{
                "text": response_text
            }]
        }
    except Exception as e:
        raise RuntimeError(
            f"Error with local model: {str(e)}\n"
            f"Try using a different model by setting 'local_model' in config.json\n"
            f"Or use 'openai' as llm_provider instead."
        ) from e


def query_bedrock_sonet(prompt):
    """
    Query the Bedrock SONET model with the provided prompt (optional, requires AWS).

    Args:
        prompt (str): The prompt to query the Bedrock SONET model.

    Returns:
        dict: Response from the Bedrock SONET model.
    """
    if not BOTO3_AVAILABLE:
        raise ImportError("boto3 not installed. Install with: pip install boto3")
    
    try:
        aws_credentials = load_config_data(config_file_path)
        region = aws_credentials.get('region', 'us-east-1')

        bedrock = boto3.client(
            service_name="bedrock-runtime", region_name=region
        )
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

        response = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4086,
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }],
            }),
        )

        result = json.loads(response.get("body").read())
        return result
    except Exception as e:
        error_msg = str(e)
        if "Unable to locate credentials" in error_msg or "NoCredentialsError" in str(type(e).__name__):
            raise RuntimeError(
                "AWS credentials not found. Please configure AWS or use a different LLM provider.\n"
                "Set 'llm_provider' in config.json to 'openai' or 'local' instead."
            ) from e
        else:
            raise RuntimeError(f"Error connecting to AWS Bedrock: {error_msg}") from e


def query_llm(prompt):
    """
    Query the configured LLM provider with the provided prompt.
    This is the main function that routes to the appropriate provider.

    Args:
        prompt (str): The prompt to query the LLM.

    Returns:
        dict: Response from the LLM in a standardized format.
    """
    config = load_config_data(config_file_path)
    provider = config.get('llm_provider', 'openai').lower()
    
    if provider == 'openai':
        return query_openai(prompt)
    elif provider == 'local':
        model_name = config.get('local_model', 'gpt2')
        return query_local_model(prompt, model_name)
    elif provider == 'bedrock' or provider == 'aws':
        return query_bedrock_sonet(prompt)
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}\n"
            "Set 'llm_provider' in config.json to one of: 'openai', 'local', or 'bedrock'"
        )

