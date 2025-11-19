"""
Speech Processing and Post-Processing Module

This module provides functions to process audio files, generate transcripts, and perform post-processing on the transcripts.

Functions:
    generate_transcript_from_file(file_path, soap=False, birp=False, instructions_file=None):
        Generate transcript from audio file and perform SOAP and/or BIRP post-processing.

    generate_soap_post_processing(transcript, instruction_file_path):
        Perform SOAP post-processing on the transcript.

    generate_birp_post_processing(transcript, instruction_file_path):
        Perform BIRP post-processing on the transcript.

    get_transcript(audio_path):
        Extract transcript from audio file using Automatic Speech Recognition (ASR).
"""

import os
import time
import subprocess
import torch
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

import modules.utils
from modules.model import query_llm
from modules.prompt_generator import generate_custom_birp_prompt, generate_custom_soap_prompt
from modules.utils import save_file

# Try to import soundfile for WAV files (doesn't require ffmpeg)
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    try:
        from scipy.io import wavfile
        SCIPY_AVAILABLE = True
    except ImportError:
        SCIPY_AVAILABLE = False

# Try to import resampling libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    try:
        from scipy import signal
        SCIPY_SIGNAL_AVAILABLE = True
    except ImportError:
        SCIPY_SIGNAL_AVAILABLE = False

project_root = os.path.dirname(os.path.dirname(__file__))  # Navigate 2 levels up from the script directory
config_file_path = os.path.join(project_root, 'config.json')


def generate_soap_post_processing(transcript, instruction_file_path):
    """
    Perform SOAP post-processing on the given transcript.

    Args:
        transcript (str): The transcript to be processed.
        instruction_file_path (str): Path to the instruction file (optional).

    Returns:
        dict: SOAP response generated based on the transcript.
    """

    prompt = generate_custom_soap_prompt(transcript, instruction_file_path)
    soap_response = query_llm(prompt)
    print("SOAP note successfully generated....")

    return soap_response


def generate_birp_post_processing(transcript, instruction_file_path):
    """
    Perform BIRP post-processing on the given transcript.

    Args:
        transcript (str): The transcript to be processed.
        instruction_file_path (str): Path to the instruction file (optional).

    Returns:
        dict: BIRP response generated based on the transcript.
    """
    prompt = generate_custom_birp_prompt(transcript, instruction_file_path)
    birp_response = query_llm(prompt)
    print("BIRP note successfully generated....")

    return birp_response



def load_audio_file(audio_path):
    """
    Load audio file and return audio array and sample rate.
    Uses soundfile for WAV files (no ffmpeg required), falls back to other methods.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        tuple: (audio_array, sample_rate) where audio_array is numpy array and sample_rate is int.
    """
    file_ext = os.path.splitext(audio_path)[1].lower()
    
    # For WAV files, use soundfile (doesn't require ffmpeg)
    if file_ext == '.wav':
        # Check for soundfile at runtime (in case it was installed after module import)
        soundfile_available = SOUNDFILE_AVAILABLE
        soundfile_sf = None
        if not soundfile_available:
            try:
                import soundfile as soundfile_sf
                soundfile_available = True
            except ImportError:
                soundfile_available = False
        
        if soundfile_available:
            # Use runtime-imported sf if available, otherwise use module-level sf
            if soundfile_sf is not None:
                audio_array, sample_rate = soundfile_sf.read(audio_path)
            else:
                audio_array, sample_rate = sf.read(audio_path)
            # Ensure mono (single channel) and float32 format
            if len(audio_array.shape) > 1:
                # Convert stereo to mono by taking the mean
                audio_array = np.mean(audio_array, axis=1)
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            return audio_array, sample_rate
        elif SCIPY_AVAILABLE:
            sample_rate, audio_array = wavfile.read(audio_path)
            # Ensure mono (single channel)
            if len(audio_array.shape) > 1:
                # Convert stereo to mono by taking the mean
                audio_array = np.mean(audio_array, axis=1)
            # Convert to float32 and normalize if needed
            if audio_array.dtype == np.int16:
                audio_array = audio_array.astype(np.float32) / 32768.0
            elif audio_array.dtype == np.int32:
                audio_array = audio_array.astype(np.float32) / 2147483648.0
            elif audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            return audio_array, sample_rate
        else:
            raise ImportError(
                "Neither soundfile nor scipy is installed. "
                "Please install one of them: pip install soundfile or pip install scipy"
            )
    else:
        # For other formats, try to use the file path directly
        # This will require ffmpeg, but we'll let the pipeline handle the error
        return None, None


def get_transcript(audio_path):
    """
    Extract transcript from audio file using Automatic Speech Recognition (ASR).

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        str: Transcript extracted from the audio file.
    """
    print(f"Input file - {audio_path}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    config = modules.load_config_data(config_file_path)
    model_name = config['model_choice']
    model_id = model_name

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Try to load audio file directly (for WAV files, avoids ffmpeg requirement)
    file_ext = os.path.splitext(audio_path)[1].lower()
    
    # Verify soundfile is available at runtime (in case it was installed after module import)
    soundfile_available = SOUNDFILE_AVAILABLE
    if not soundfile_available:
        try:
            import soundfile as sf
            soundfile_available = True
        except ImportError:
            soundfile_available = False
    
    if file_ext == '.wav' and soundfile_available:
        print(f"Loading WAV file using soundfile (no ffmpeg required)...")
        try:
            # Load WAV file using soundfile (no ffmpeg needed)
            audio_array, sample_rate = load_audio_file(audio_path)
            if audio_array is not None and sample_rate is not None:
                # Resample to 16000 Hz if needed (Whisper models require 16kHz)
                target_sample_rate = 16000
                if sample_rate != target_sample_rate:
                    print(f"Resampling audio from {sample_rate} Hz to {target_sample_rate} Hz...")
                    # Try librosa first (best quality), then scipy.signal
                    if LIBROSA_AVAILABLE:
                        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=target_sample_rate)
                        sample_rate = target_sample_rate
                        # Ensure float32 format after resampling
                        if audio_array.dtype != np.float32:
                            audio_array = audio_array.astype(np.float32)
                    elif SCIPY_SIGNAL_AVAILABLE:
                        num_samples = int(len(audio_array) * target_sample_rate / sample_rate)
                        audio_array = signal.resample(audio_array, num_samples)
                        sample_rate = target_sample_rate
                        # Ensure float32 format after resampling
                        if audio_array.dtype != np.float32:
                            audio_array = audio_array.astype(np.float32)
                    else:
                        # If no resampling library available, let processor handle it
                        # but this might fail, so warn the user
                        print(f"Warning: No resampling library available. Audio is {sample_rate} Hz, model expects {target_sample_rate} Hz.")
                
                # Process audio using the processor (handles sample rate conversion)
                inputs = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
                # Move inputs to the correct device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                # Generate transcription using the model directly
                # For Whisper models, language can be set via forced_decoder_ids or language parameter
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs, 
                        max_new_tokens=128,
                        language="english"
                    )
                # Decode the transcription
                transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                result = {"text": transcription}
            else:
                raise ValueError("Failed to load audio array from file")
        except Exception as e:
            # For WAV files with soundfile available, don't fall back to pipeline
            # Raise a more informative error instead
            raise RuntimeError(
                f"Failed to process WAV file with soundfile. Error: {str(e)}\n"
                f"File: {audio_path}\n"
                "Please check that the file is a valid WAV file and try again."
            ) from e
    else:
        # For non-WAV files or if soundfile is not available, use pipeline
        # This requires ffmpeg for non-WAV formats
        if file_ext != '.wav':
            print(f"Note: {file_ext} format requires ffmpeg. Attempting to load...")
        elif not SOUNDFILE_AVAILABLE:
            print("Note: soundfile not available, using pipeline (requires ffmpeg for WAV files)...")
        try:
            result = pipe(audio_path, generate_kwargs={"language": "english"})
        except Exception as e:
            if "ffmpeg" in str(e).lower():
                raise RuntimeError(
                    f"Failed to load audio file. Error: {str(e)}\n"
                    f"For WAV files, ensure soundfile is installed: pip install soundfile\n"
                    f"For other formats ({file_ext}), install ffmpeg: https://ffmpeg.org/download.html"
                ) from e
            else:
                raise

    print("transcript successfully generated....")

    return result["text"]


def generate_transcript_from_file(file_path, soap=False, birp=False, instructions_file=None):
    """
    Generate transcript from audio file and perform SOAP and/or BIRP post-processing.

    Args:
        file_path (str): Path to the audio file.
        soap (bool): Perform SOAP post-processing if True (default is False).
        birp (bool): Perform BIRP post-processing if True (default is False).
        instructions_file (str): Path to the instruction file for post-processing (optional).

    Returns:
        str: Transcript extracted from the audio file.
    """
    # Start timer
    start_time = time.time()

    # Step1: Generate transcript
    transcript = get_transcript(file_path)

    # Step2: Apply SOAP, BIRP, or both post-processing steps if requested
    if soap:
        soap_notes = generate_soap_post_processing(transcript, instructions_file)
        soap_text = soap_notes['content'][0]['text']  # Extract the 'text' content from the dictionary
        save_file(soap_text, '')
        print("SOAP note saved....")

    if birp:
        birp_notes = generate_birp_post_processing(transcript, instructions_file)
        birp_text = birp_notes['content'][0]['text']  # Extract the 'text' content from the dictionary
        save_file(birp_text, '')
        print("BIRP note saved....")

    # End timer
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Process completed in {elapsed_time:.2f} seconds.")
    return transcript

