import gradio as gr
from coverletter_generator import (
    load_and_split_documents,
    create_vectorstore,
    create_qa_chain,
    generate_cover_letter,
)

def generate_cover_letter_with_ui(cv, transcript, certificates, job_description, temperature):
    """Generates a cover letter using the provided documents and job description."""
    # Only CV is compulsory, handle optional documents
    transcript_path = transcript.name if transcript else None
    certificates_path = certificates.name if certificates else None
    splits = load_and_split_documents(cv.name, transcript_path, certificates_path)
    vectorstore = create_vectorstore(splits)
    qa_chain = create_qa_chain(vectorstore=vectorstore, temperature=temperature)
    cover_letter = generate_cover_letter(job_description, qa_chain)
    return cover_letter

iface = gr.Interface(
    fn=generate_cover_letter_with_ui,
    inputs=[
        gr.File(label="Upload your CV (docx)", file_types=['.docx']),  # CV is required
        gr.File(label="Upload your transcripts (docx) (Optional)", file_types=['.docx']),
        gr.File(label="Upload your certificates (docx) (Optional)", file_types=['.docx']),
        gr.Textbox(lines=10, label="Enter job description"),
        gr.Slider(0, 1, value=0.7, label="Temperature"),
    ],
    outputs="text",
    title="Cover Letter Generator",
    description="Generate a cover letter based on your CV, transcripts, and certificates.",
)

iface.launch()