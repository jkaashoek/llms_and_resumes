from PyPDF2 import PdfReader
from edsl import  Agent, Model
from edsl.questions import QuestionFreeText
import os

class TextObj():
    def __init__(self, fp) -> None:
        self.fp = fp
        return
    
    def llm_clean_text(self, text, model = 'gpt-4-1106-preview'):
        '''
        Cleans text using an LLM model
        '''
        agent = Agent(traits={'role': 'improver', 'persona': 'You are an expert in formatting text, specifically resumes and job descriptions.'})
        question = QuestionFreeText(question_name = 'llm_clean', question_text = 'Nicely format the following text. Do not change any of the details, only fix spelling or grammar mistakes and fix formatting issues. Output only the cleaned text.\n\n' + text)
        edsl_model = Model(model)
        res = question.by(agent).by(edsl_model).run()
        ans = res.select('llm_clean').to_list()[0]
        return ans
    
    def extract_text(self, clean_text = False):
        '''
        Extracts text from a text object
        '''
        filename = os.fsdecode(self.fp)
        if filename.endswith(".pdf"):
            reader = PdfReader(self.fp)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'

        elif filename.endswith('.txt'):
            with open(self.fp, 'r') as file:
                text = file.read()
        
        self.text = text
        self.cleaned_text = self.llm_clean_text(text) if clean_text else None

        return

    def summarize(self, model = 'gpt-4-1106-preview', person_instructions = None):
        '''
        Summarizes the text object
        '''
        if person_instructions is None:
            person_instructions = 'You are an expert in summarizing text, specifically resumes and job descriptions.'
        agent = Agent(traits={'role': 'improver', 'persona': person_instructions})
        question = QuestionFreeText(question_name = 'llm_clean', question_text = 'Nicely format the following text. Do not change any of the details, only fix spelling or grammar mistakes and fix formatting issues. Output only the cleaned text.\n\n' + text)
        edsl_model = Model(model)
        res = question.by(agent).by(edsl_model).run()
        ans = res.select('llm_clean').to_list()[0]
        self.text_summary = ans
        return ans

class Resume(TextObj):
    def __init__(self, resume_path) -> None:
        super().__init__(resume_path)

    def extract_text(self, clean_text = False):
        return super().extract_text(clean_text)

    def summarize(self):
        persona_instructions = 'You are an expert recruiter who has been hired to summarize resumes for hiring managers to make decisions on who to hire.'
        super().summarize(person_instructions = persona_instructions)

    def modify_resume(self, job_description : JobDescription):
        '''
        Modify a resume
        '''
        pass

class JobDescription(TextObj):
    def __init__(self, job_description_path) -> None:
        super().__init__(job_description_path)

    def extract_text(self):
        return super().extract_text()

    def summarize(self):
        persona_instructions = 'You are an expert recruiter who has been hired to summarize job descriptions for hiring managers to make decisions on who to hire.'
        super().summarize(person_instructions = persona_instructions)

    def evaluate_resume(self, resume : Resume):
        '''
        Evaluates a resume against a job description
        '''
        pass

    def cut_resumes(self, resume : list[Resume]):
        '''
        Cut resumes to fit a job description
        '''
        pass

    def select_resumes(self, resumes : list[Resume]):
        '''
        Selects resumes that fit a job description
        '''
        pass
