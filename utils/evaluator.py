def get_questions(client, context):
    
    prompt = f"""
    You are a Senior Technical Interviewer.

    Candidate Resume:
    {context}

    Analyze the resume carefully.

    Generate:

    1. Five personalized Technical Interview Questions.
    2. Five personalized HR Interview Questions.

    Requirements:
    - Questions must be based on the candidate's projects.
    - Mention technologies found in the resume.
    - Mention Machine Learning, Python, SQL, AI, Deep Learning if present.
    - Mention internships if present.
    - Do NOT generate generic questions.
    - Ask real interview-style questions.
    - Format clearly with numbering.

    Example:
    Instead of asking:
    "What is Machine Learning?"

    Ask:
    "In your AI-Powered Virtual Fashion Designer project, why did you choose Diffusion Models over GANs?"
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


def evaluate_answer(client, question, answer):

    prompt = f"""
    Interview Question:
    {question}

    Candidate Answer:
    {answer}

    Evaluate professionally.

    Give:

    Overall Score: /10

    Technical Knowledge:
    Communication Skills:
    Confidence Level:

    Strengths:
    - Point 1
    - Point 2

    Areas for Improvement:
    - Point 1
    - Point 2

    Suggested Better Answer:
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content