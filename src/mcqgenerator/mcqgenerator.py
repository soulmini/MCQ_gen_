import os
import json
import pandas as pd
from dotenv import load_dotenv
# Import the updated Together class
from langchain_together import Together
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

load_dotenv()

api_key = os.getenv("TOGETHER_API_KEY")

# Initialize the model
model = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.7,
    max_tokens=128,
    top_k=50,
    together_api_key=api_key
)

# Print the model to confirm initialization
# print(model)

RESPONSE_JSON = {
    "1": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "2": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "3": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
}


TEMPLATE="""
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}
"""


quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=TEMPLATE
    )


quiz_chain=LLMChain(llm=model, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)



TEMPLATE2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt=PromptTemplate(input_variables=["subject", "quiz"], template=TEMPLATE2)

review_chain=LLMChain(llm=model, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)



generate_evaluate_chain=SequentialChain(chains=[quiz_chain, review_chain], input_variables=["text", "number", "subject", "tone", "response_json"],
                                        output_variables=["quiz", "review"], verbose=True,)


file_path = "/home/ayush/Ayush/genAiProject/ref.txt"

with open(file_path, 'r') as file :
    Text = file.read()


# print(Text)

NUMBER=5 
SUBJECT="biology"
TONE="simple"


response=generate_evaluate_chain(
    {
        "text": Text,
        "number": NUMBER,
        "subject":SUBJECT,
        "tone": TONE,
        "response_json": json.dumps(RESPONSE_JSON)
    }
)

print(response["quiz"])