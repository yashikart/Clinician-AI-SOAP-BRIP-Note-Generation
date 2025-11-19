# Clinical Note Generation from Doctor-Patient Conversations using LLM

This project uses open-source LLMs to generate clinical notes from doctor–patient conversations. It includes note generation using models like MedAlpaca, evaluation with BLEU, ROUGE, and BERTScore, and analysis to improve accuracy and faithfulness.

Additionally, this Python project utilizes OpenAI's Whisper speech recognition model to generate transcripts from audio or video files, and then leverages Anthropic's Sonnet LLM (Large Language Model) or OpenAI API to generate structured medical notes such as SOAP (Subjective, Objective, Assessment, Plan) or BIRP (Behavior, Intervention, Response, Plan) notes based on the transcripts.

## Features

- Generate clinical notes from doctor-patient dialogues using open-source LLMs (MedAlpaca, Mistral)
- Evaluate generated notes using BLEU, ROUGE, and BERTScore metrics
- Generate transcripts from audio or video files using Whisper
- Generate SOAP or BIRP notes from the transcripts using OpenAI API or local models
- Customize prompts and templates for different note formats
- Save generated notes as text files
- Streamlit web interface for easy file upload and note generation
- Fine-tune models using LoRA for improved performance

## Requirements

- Python 3.10 or higher
- OpenAI Whisper (for speech recognition) via HuggingFace
- OpenAI API or local models (for text generation)
- Additional Python libraries: `boto3`, `transformers`, `torch`, `accelerate`, `soundfile`, `librosa`, `openai`, `streamlit`

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yashikart/Clinician-AI-API-SOAP-Brief-Note-Generation.git
cd Clinician-AI-API-SOAP-Brief-Note-Generation
```

2. Install the required Python libraries:

```bash
pip install -r requirements.txt
```

3. Configure the LLM provider in `config.json`:
   - Set `llm_provider` to `"openai"` (requires API key), `"local"` (free, runs locally), or `"bedrock"` (requires AWS)

4. For OpenAI (recommended):
   - Get your API key from https://platform.openai.com/api-keys
   - Set it as an environment variable:
     ```powershell
     # Windows PowerShell
     $env:OPENAI_API_KEY = "your-api-key-here"
     ```

## Usage

### Command Line Interface

Run the main script:

```bash
python run.py
```

Follow the prompts to:
1. Enter the path to your audio file
2. Choose whether to generate SOAP notes (yes/no)
3. Choose whether to generate BIRP notes (yes/no)
4. Optionally provide an instruction file

### Streamlit Web Interface

Run the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser where you can:
- Upload WAV files or select from test files
- Configure SOAP/BIRP note generation
- View and download generated notes

See `STREAMLIT_README.md` for detailed instructions.

## Configuration

Edit `config.json` to configure:
- `model_choice`: Whisper model for transcription (default: "openai/whisper-small")
- `llm_provider`: LLM provider for note generation ("openai", "local", or "bedrock")
- `local_model`: Model name if using local provider (default: "gpt2")

## Supported Audio Formats

- WAV (recommended - no ffmpeg required)
- MP3, M4A, WebM, MP4, MPGA, MPEG, AVI, FLV, MOV, WMV (requires ffmpeg)

## Project Structure

```
.
├── app.py                      # Streamlit web application
├── run.py                       # Command-line interface
├── config.json                  # Configuration file
├── requirements.txt             # Python dependencies
├── modules/
│   ├── __init__.py
│   ├── model.py                 # LLM provider module (OpenAI, local, Bedrock)
│   ├── transcript_generator.py  # Audio transcription and note generation
│   ├── prompt_generator.py      # SOAP/BIRP prompt templates
│   └── utils.py                 # Utility functions
├── test/                        # Test audio files
└── README.md                    # This file
```

## License

See LICENSE.md for details.
