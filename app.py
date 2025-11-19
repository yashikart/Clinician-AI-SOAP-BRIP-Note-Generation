"""
Streamlit App for Clinical Note Generation
Upload WAV files or select from test files to generate SOAP and BIRP notes
"""

import streamlit as st
import os
import tempfile
from modules.transcript_generator import generate_transcript_from_file, get_transcript
from modules.transcript_generator import generate_soap_post_processing, generate_birp_post_processing

# Page configuration
st.set_page_config(
    page_title="Clinical Note Generator",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("🏥 Clinical Note Generator")
st.markdown("Upload a WAV audio file or select from test files to generate SOAP and BIRP notes from doctor-patient conversations.")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    generate_soap = st.checkbox("Generate SOAP Notes", value=True)
    generate_birp = st.checkbox("Generate BIRP Notes", value=True)
    
    instruction_file = st.file_uploader(
        "Upload Instruction File (Optional)",
        type=['txt'],
        help="Upload a text file with custom instructions for note generation"
    )
    
    st.markdown("---")
    st.markdown("### 📋 About")
    st.markdown("""
    This app generates clinical notes from audio transcripts:
    - **SOAP**: Subjective, Objective, Assessment, Plan
    - **BIRP**: Behavior, Intervention, Response, Plan
    """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📁 File Selection")
    
    # Option to use test files
    use_test_file = st.checkbox("Use Test File", value=False)
    
    if use_test_file:
        # Get test files from test folder
        test_folder = "test"
        test_files = []
        
        if os.path.exists(test_folder):
            all_files = [f for f in os.listdir(test_folder) if f.endswith('.wav')]
            # Select 5 test files (first 5 alphabetically)
            test_files = sorted(all_files)[:5]
        
        if test_files:
            st.markdown("**Available Test Files:**")
            for i, test_file in enumerate(test_files, 1):
                st.text(f"{i}. {test_file}")
            
            selected_test_file = st.selectbox(
                "Select a test file:",
                test_files,
                help="Choose from available test files",
                index=0
            )
            selected_file_path = os.path.join(test_folder, selected_test_file)
            st.success(f"✅ Selected: **{selected_test_file}**")
        else:
            st.warning("No test files found in the 'test' folder.")
            selected_file_path = None
            use_test_file = False
    else:
        selected_file_path = None
    
    st.markdown("---")
    st.markdown("**Or upload your own file:**")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a WAV file",
        type=['wav', 'mp3', 'm4a', 'webm', 'mp4'],
        help="Upload an audio file containing doctor-patient conversation",
        label_visibility="collapsed"
    )
    
    # Determine which file to use (test file takes priority)
    if use_test_file and selected_file_path and os.path.exists(selected_file_path):
        file_to_process = selected_file_path
        file_name = os.path.basename(selected_file_path)
    elif uploaded_file is not None:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_to_process = tmp_file.name
            file_name = uploaded_file.name
    else:
        file_to_process = None
        file_name = None

with col2:
    st.header("📊 Status")
    
    if file_to_process:
        st.success(f"✅ File ready: **{file_name}**")
        
        # Process button
        if st.button("🚀 Generate Notes", type="primary", use_container_width=True):
            if not generate_soap and not generate_birp:
                st.warning("Please select at least one note type (SOAP or BIRP)")
            else:
                # Process the file
                with st.spinner("Processing audio file... This may take a few minutes."):
                    try:
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Step 1: Generate transcript
                        status_text.text("Step 1/3: Generating transcript from audio...")
                        progress_bar.progress(20)
                        
                        transcript = get_transcript(file_to_process)
                        progress_bar.progress(50)
                        
                        # Display transcript
                        st.session_state['transcript'] = transcript
                        st.session_state['soap_note'] = None
                        st.session_state['birp_note'] = None
                        
                        # Step 2: Generate SOAP notes if requested
                        if generate_soap:
                            status_text.text("Step 2/3: Generating SOAP notes...")
                            progress_bar.progress(70)
                            
                            instruction_path = None
                            if instruction_file:
                                # Save instruction file temporarily
                                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_inst:
                                    tmp_inst.write(instruction_file.read().decode('utf-8'))
                                    instruction_path = tmp_inst.name
                            
                            soap_response = generate_soap_post_processing(transcript, instruction_path)
                            soap_text = soap_response['content'][0]['text']
                            st.session_state['soap_note'] = soap_text
                            
                            if instruction_path and os.path.exists(instruction_path):
                                os.unlink(instruction_path)
                        
                        # Step 3: Generate BIRP notes if requested
                        if generate_birp:
                            status_text.text("Step 3/3: Generating BIRP notes...")
                            progress_bar.progress(85)
                            
                            instruction_path = None
                            if instruction_file:
                                # Save instruction file temporarily
                                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_inst:
                                    instruction_file.seek(0)  # Reset file pointer
                                    tmp_inst.write(instruction_file.read().decode('utf-8'))
                                    instruction_path = tmp_inst.name
                            
                            birp_response = generate_birp_post_processing(transcript, instruction_path)
                            birp_text = birp_response['content'][0]['text']
                            st.session_state['birp_note'] = birp_text
                            
                            if instruction_path and os.path.exists(instruction_path):
                                os.unlink(instruction_path)
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Processing complete!")
                        st.success("Notes generated successfully!")
                        
                        # Clean up temporary file if it was uploaded
                        if uploaded_file and os.path.exists(file_to_process):
                            os.unlink(file_to_process)
                            
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)
                        # Clean up temporary file
                        if uploaded_file and os.path.exists(file_to_process):
                            os.unlink(file_to_process)
    else:
        st.info("👆 Please upload a file or select a test file to begin")

# Display results
if 'transcript' in st.session_state and st.session_state['transcript']:
    st.markdown("---")
    st.header("📝 Results")
    
    # SOAP Notes section
    if generate_soap and st.session_state.get('soap_note'):
        st.markdown("### 📋 SOAP Notes")
        st.text_area(
            "SOAP Note:",
            value=st.session_state['soap_note'],
            height=400,
            disabled=True,
            label_visibility="collapsed"
        )
        st.download_button(
            "Download SOAP Note",
            data=st.session_state['soap_note'],
            file_name=f"SOAP_{file_name.replace('.wav', '')}.txt",
            mime="text/plain",
            key="download_soap"
        )
    
    # BIRP Notes section
    if generate_birp and st.session_state.get('birp_note'):
        st.markdown("### 📋 BIRP Notes")
        st.text_area(
            "BIRP Note:",
            value=st.session_state['birp_note'],
            height=400,
            disabled=True,
            label_visibility="collapsed"
        )
        st.download_button(
            "Download BIRP Note",
            data=st.session_state['birp_note'],
            file_name=f"BIRP_{file_name.replace('.wav', '')}.txt",
            mime="text/plain",
            key="download_birp"
        )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Clinical Note Generator | Powered by Whisper & OpenAI</div>",
    unsafe_allow_html=True
)

