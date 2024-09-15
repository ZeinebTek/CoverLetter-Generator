# AI-Powered Cover Letter Generator

This project uses Google's Gemini Pro model and LangChain to generate personalized cover letters based on your CV, transcripts, and certificates. It leverages Retrieval Augmented Generation (RAG) to extract relevant information from your documents and tailor the cover letter to the specific job description you provide.

## Features

- **Personalized Cover Letters:** Generates cover letters tailored to the specific job description you provide.
- **Retrieval Augmented Generation (RAG):** Uses RAG to extract relevant information from your CV, transcripts, and certificates.
- **Google Gemini Pro:** Leverages the power of Google's advanced language model for high-quality text generation.
- **Gradio Interface:** Provides a user-friendly interface to upload your documents, enter the job description, and generate the cover letter.
- **Temperature Control:** Allows you to adjust the creativity and randomness of the generated cover letter using a slider.

## Requirements

- Python 3.9+

## Installation

### **Clone the repository:**

   ```bash
   git clone https://github.com/ZeinebTek/CoverLetter-Generator.git
   ```

### **Install the required libraries:**


    cd CoverLetter-Generator
    pip install -r requirements.txt
    
   

### **Prepare the .env file:**

- Create a file named .env in the project directory.
- Add your Google Cloud API key to the .env file in the following format:

    ```bash
    GOOGLE_API_KEY="your key"
    ```

## Usage

### Run the Gradio demo:


    python demo_gradio.py

### Upload your documents:
 * Upload your CV (docx or pdf format).
 * Optionally upload your transcripts and certificates (docx or pdf format).
### Enter the job description:
 * Paste the job description into the text box.
### Adjust the temperature (optional):
 * Use the slider to control the creativity of the generated cover letter.
### Generate the cover letter:
Click the "Submit" button to generate the cover letter.

## File Structure
### coverletter_generator.py: 
Contains the core functions for loading documents, creating the vectorstore, building the QA chain, and generating the cover letter.
### demo_gradio.py: 
Creates the Gradio interface for the cover letter generator.
### requirements.txt: 
Lists the required Python libraries.

## Contributing
Contributions are welcome! Please feel free to open issues or pull requests.
