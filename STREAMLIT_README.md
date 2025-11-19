# Streamlit App - Clinical Note Generator

## 🚀 Quick Start

1. **Install dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

2. **Set OpenAI API Key** (if using OpenAI):
   ```powershell
   # Windows PowerShell
   $env:OPENAI_API_KEY = "your-api-key-here"
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** - The app will automatically open at `http://localhost:8501`

## 📋 Features

- **Upload WAV files** - Upload your own audio files
- **Test Files** - Select from 5 pre-loaded test files in the `test/` folder
- **Generate SOAP Notes** - Subjective, Objective, Assessment, Plan
- **Generate BIRP Notes** - Behavior, Intervention, Response, Plan
- **Download Results** - Download transcripts and notes as text files
- **Custom Instructions** - Upload custom instruction files for note generation

## 🎯 Usage

1. **Select File Source**:
   - Check "Use Test File" to select from test files
   - Or upload your own WAV file

2. **Configure Options**:
   - Check/uncheck "Generate SOAP Notes"
   - Check/uncheck "Generate BIRP Notes"
   - Optionally upload an instruction file

3. **Generate Notes**:
   - Click "🚀 Generate Notes" button
   - Wait for processing (may take a few minutes)

4. **View & Download**:
   - View SOAP and BIRP notes
   - Download any of the results as text files

## 📁 Test Files

The app automatically shows the first 5 WAV files from the `test/` folder alphabetically.

## ⚙️ Configuration

The app uses the same `config.json` file as the command-line version:
- Set `llm_provider` to `"openai"`, `"local"`, or `"bedrock"`
- For OpenAI, set the `OPENAI_API_KEY` environment variable

## 🐛 Troubleshooting

- **"Unable to locate credentials"**: Set your OpenAI API key or change LLM provider in config.json
- **File not found**: Make sure test files exist in the `test/` folder
- **Processing takes long**: Audio transcription and note generation can take several minutes depending on file size

