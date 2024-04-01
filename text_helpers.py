from PyPDF2 import PdfReader
from edsl import  Agent, Model, Survey
from edsl.questions import QuestionFreeText, QuestionLinearScale
import os

class TextObj():
    def __init__(self, fp) -> None:
        self.fp = fp
        self.extract_text()
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
        self.cleaned_text = self.llm_clean_text(text) if clean_text else text

        return self.text, self.cleaned_text
    

    def summarize(self, model = 'gpt-4-1106-preview', person_instructions = None):
        '''
        Summarizes the text object
        '''
        if person_instructions is None:
            person_instructions = 'You are an expert in summarizing text, specifically resumes and job descriptions.'
        agent = Agent(traits={'role': 'improver', 'persona': person_instructions})
        question = QuestionFreeText(question_name = 'summarize', question_text = 'Summarize the following text.\n\n' + text)
        edsl_model = Model(model)
        res = question.by(agent).by(edsl_model).run()
        ans = res.select('summarize').to_list()[0]
        self.text_summary = ans
        return ans
    
    def __str__(self) -> str:
        return self.text

class Resume(TextObj):
    def __init__(self, resume_path) -> None:
        super().__init__(resume_path)

    def extract_text(self, clean_text = False):
        return super().extract_text(clean_text)

    def summarize(self):
        persona_instructions = 'You are an expert recruiter who has been hired to summarize resumes for hiring managers to make decisions on who to hire.'
        super().summarize(person_instructions = persona_instructions)

    def modify_resume(self, agent_instructions : str, model = 'gpt-4-1106-preview'):
        '''
        Modify a resume
        '''
        agent = Agent(traits={'role': 'improver', 'persona': agent_instructions})
        question = QuestionFreeText(question_name = 'modify', question_text = 'Modify the following resume.\n\n' + self.cleaned_text)
        edsl_model = Model(model)
        res = question.by(agent).by(edsl_model).run()
        return 
class JobDescription(TextObj):
    def __init__(self, job_description_path) -> None:
        super().__init__(job_description_path) 

    def extract_text(self):
        return super().extract_text()

    def summarize(self):
        persona_instructions = 'You are an expert recruiter who has been hired to summarize job descriptions for hiring managers to make decisions on who to hire.'
        super().summarize(person_instructions = persona_instructions)

    # def evaluate_resume(self, resume : Resume):
    #     '''
    #     Evaluates a resume against a job description
    #     '''
    #     pass

    # def cut_resumes(self, resume : list[Resume]):
    #     '''
    #     Cut resumes to fit a job description
    #     '''
    #     pass

    # def select_resumes(self, resumes : list[Resume]):
    #     '''
    #     Selects resumes that fit a job description
    #     '''
    #     pass
        

## TODO 
# These should all return just the prompts
# They should be wrapped in something like a Pool object that contains all of the text objects
# We can then play around with evaluating a pool against a single text object and whatnot
# What if we actually want to run the prompts on the individual text objects?
# We could have a run method that just takes the right prompts and runs them?
    
class TextPool():
    def __init__(self, fp) -> None:
        self.fp = fp
        # TODO: Extract the texts
        self.texts = None
        return

    # Now evaluate the texts against some other text
    

def evaluate_texts(text1 : str, text2 : list[str], model = 'gpt-4-1106-preview', eval_options = {}):
    '''
    Evaluates two texts
    '''
    agent_instructions = eval_options.get('agent_instructions', 'You are a recruiter who is in an expert in screening resumes. You have been hired to evaluate resumes for a job opening.')
    agent = Agent(traits={'role': 'evaluator', 'persona': agent_instructions})
    question = Survey(questions = [QuestionLinearScale(
        question_name = f'evaluate_{i}', 
        question_text = 'Evaluate the following two texts on a scale from 1 to 10 with 1 being the worst and 10 being the best\n\n' + t,
        question_options = list(range(1, 11))) for i, t in enumerate(text2)])
    edsl_model = Model(model)
    res = question.by(agent).by(edsl_model).run()
    # ans = res.select('answer.*').to_list()[0]
    return res


# Example usage
resume = Resume('resumes/business_resume.pdf')
# print("no cleaning")
text, cleaned = resume.extract_text(True)
# print(resume.extract_text(False))
# print("with cleaning")
# print(resume.extract_text(True))
with open('resumes/business_resume_cleaned.txt', 'w') as f:
    f.write(cleaned)

with open('resumes/business_resume_not_cleaned.txt', 'w') as f:
    f.write(text)

print("summarized")
summ = resume.summarize()
print(summ)
print(resume.text_summary)