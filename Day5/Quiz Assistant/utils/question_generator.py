from utils.summarizer import model

def generate_mcqs(summary, num_questions=2):
    prompt = f"""Based on the following summary, generate {num_questions} multiple-choice questions with four options each, and clearly label the correct answer for each question:\n\n{summary}"""
    response = model.generate_content(prompt)
    return response.text
