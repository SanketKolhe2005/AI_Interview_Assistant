import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq

from utils.pdf_reader import extract_text
from utils.embeddings import (
    chunk_text,
    create_vector_store,
    retrieve
)
from utils.evaluator import (
    get_questions,
    evaluate_answer
)

# Load Environment Variables
load_dotenv()

# Groq Client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Streamlit Config
st.set_page_config(
    page_title="AI Interview Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Interview Assistant")
st.write("Upload your resume and get personalized interview questions.")

# Role Selection
role = st.selectbox(
    "Select Job Role",
    [
        "Machine Learning Engineer",
        "Data Analyst",
        "AI Engineer",
        "Python Developer"
    ]
)

# Upload Resume
uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

if uploaded_file:

    with st.spinner("Reading Resume..."):
        resume_text = extract_text(uploaded_file)

    chunks = chunk_text(resume_text)

    index = create_vector_store(chunks)

    st.success("✅ Resume Processed Successfully!")

    if st.button("Generate Interview Questions"):

        with st.spinner("Generating Questions..."):

            context = retrieve(
                f"Generate interview questions for {role}",
                chunks,
                index
            )

            questions = get_questions(
                client,
                context
            )

            st.session_state.questions = questions

    if "questions" in st.session_state:

        st.subheader("📋 Interview Questions")

        st.write(st.session_state.questions)

        st.markdown("---")

        answer = st.text_area(
            "Enter Your Answer",
            height=200
        )

        if st.button("Evaluate Answer"):

            if answer.strip() == "":
                st.warning("Please enter an answer.")
            else:

                with st.spinner("Evaluating..."):

                    result = evaluate_answer(
                        client,
                        st.session_state.questions,
                        answer
                    )

                st.subheader("📊 Evaluation Report")
                st.write(result)