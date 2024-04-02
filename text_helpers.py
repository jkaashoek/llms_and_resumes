from PyPDF2 import PdfReader
from edsl import  Agent, Model, Survey
from edsl.questions import QuestionFreeText, QuestionLinearScale
import os

class TextObj():
    def __init__(self, fp, lazy_loading = True) -> None:
        self.fp = fp
        self.lazy = lazy_loading
        self.text_name = fp[:-4]
        self.text, self.cleaned_text, self.summary = None, None, None

        if not lazy_loading:
            self.text, self.cleaned_text = self.extract_text()
            question, agent = TextObj.summarize(self.cleaned_text)
            self.summary = question.by(agent).by('edsl_model').run().select(question.question_name).to_list()[0]

        return
    
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
    
    def set_summary(self, summary):
        self.summary = summary
        return

    @staticmethod
    def llm_clean_text(text_obj, person_instructions = 'You are an expert in formatting text.'):
        '''
        Cleans text using an LLM model
        '''
        agent = Agent(traits={'role': 'improver', 'persona': person_instructions})
        question = QuestionFreeText(question_name = f'llm_clean_{text_obj.text_name}', question_text = 'Nicely format the following text. Do not change any of the details, only fix spelling or grammar mistakes and fix formatting issues. Output only the cleaned text.\n\n' + text_obj.cleaned_text)
        return question, agent
    
    @staticmethod
    def summarize(text_obj, persona_instructions = 'You are an expert in summarizing text.'):
        agent = Agent(traits={'role': 'improver', 'persona': persona_instructions})
        question = QuestionFreeText(question_name = f'summarize_{text_obj.text_name}', question_text = 'Summarize the following text.\n\n' + text_obj.cleaned_text)
        return question, agent

    def __str__(self) -> str:
        return self.text

class Resume(TextObj):
    def __init__(self, resume_path) -> None:
        self.modifications = []
        super().__init__(resume_path)

    def extract_text(self, clean_text = False):
        return super().extract_text(clean_text)

    def summarize(self):
        persona_instructions = 'You are an expert recruiter who has been hired to summarize resumes for hiring managers to make decisions on who to hire.'
        return super().summarize(person_instructions = persona_instructions)
    
    def add_modification(self, modification): 
        self.modifications.append(modification)
        return

    def modify_resume(self, agent_instructions : str):
        '''
        Modify a resume
        '''
        agent = Agent(traits={'role': 'improver', 'persona': agent_instructions})
        question = QuestionFreeText(question_name = 'modify', question_text = 'Modify the following resume.\n\n' + self.cleaned_text)
        # edsl_model = Model(model)
        # res = question.by(agent).by(edsl_model).run()
        # self.modifications.append(res.select('modify').to_list()[0])
        return agent, question

class JobDescription(TextObj):
    def __init__(self, job_description_path) -> None:
        super().__init__(job_description_path) 

    def extract_text(self):
        return super().extract_text()

    def summarize(self):
        persona_instructions = 'You are an expert recruiter who has been hired to summarize job descriptions for hiring managers to make decisions on who to hire.'
        return super().summarize(person_instructions = persona_instructions)

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
    def __init__(self, fp, text_type) -> None:
        self.fp = fp
        self.text_type = text_type

        if text_type == 'resumes':
            self.texts = [Resume(f) for f in os.listdir(fp)]
        elif text_type == 'job_descriptions':
            self.texts = [JobDescription(f) for f in os.listdir(fp)]
        else:
            self.texts = [TextObj(f) for f in os.listdir(fp)]
        self.text_names = [t.text_name for t in self.texts]

        return
    
    @staticmethod
    def clean_survey(survey_res):
        '''
        Cleans a survey result
        '''
        as_df = survey_res.select("answer.*").to_pandas().melt()
        as_df['question_name'] = as_df['variable'].apply(lambda x: x.split('.')[1])
        as_df = as_df.rename(columns={'value': 'answer'})
        as_df = as_df.drop(columns=['variable'])
        as_df = as_df[['question_name', 'answer']]
        return as_df
    
    @staticmethod
    def summarize(pool_obj, model = 'gpt-4-1106-preview'):
        '''
        Summarizes each text object
        '''
        # Set up our questions as a survey
        ex_text = pool_obj.texts[0]
        question_list = [x.summarize(x.text, x.text_name)[0] for x in pool_obj.texts]
        survey = Survey(questions = question_list)
        agent = ex_text.summarize(ex_text)[1]
        edsl_model = Model(model)
        res = survey.by(agent).by(edsl_model).run()

        # Clean these up
        cleaned_res = TextPool.clean_survey(res)

        # We ran them, so might as well apply them to the acutal objects themselves.
        for q, a in cleaned_res.iterrows():
            pool_obj.texts[pool_obj.text_names.index(a['question_name'])].set_summary(a['answer'])

        return cleaned_res
    
    @staticmethod
    def evaluation(pool_obj, eval_options = {}):
        '''
        Evaluates a pool of text objects against a reference text object
        '''
        agent_instructions = eval_options.get('agent_instructions', f'You are an expert in evaluating {pool_obj.text_type}s.')
        models = eval_options.get('models', Model(['gpt-4-1106-preview']))
        reference = eval_options.get('reference', None)

        if reference is not None:
            agent_instructions += f'\n\n {reference.cleaned_text}'

        agent = Agent(traits={'role': 'evaluator', 'persona': agent_instructions})
        question_list = [QuestionLinearScale(question_name = x.text_name,
                                             question = eval_options.get('question', 'Evaluate the following text on a scale of 1 to 10') + "\n\n" + x.cleaned_text,
                                             question_options = eval_options.get('options', list(range(1, 11))) ) for x in pool_obj.texts]
        survey = Survey(questions = question_list)
        res = survey.by(agent).by(models).run()
        return TextPool.clean_survey(res)
    

    # Now evaluate the texts against some other text

# def evaluate_texts(text1 : str, text2 : list[str], model = 'gpt-4-1106-preview', eval_options = {}):
#     '''
#     Evaluates two texts
#     '''
#     agent_instructions = eval_options.get('agent_instructions', 'You are a recruiter who is in an expert in screening resumes. You have been hired to evaluate resumes for a job opening.')
#     agent = Agent(traits={'role': 'evaluator', 'persona': agent_instructions})
#     question = Survey(questions = [QuestionLinearScale(
#         question_name = f'evaluate_{i}', 
#         question_text = 'Evaluate the following two texts on a scale from 1 to 10 with 1 being the worst and 10 being the best\n\n' + t,
#         question_options = list(range(1, 11))) for i, t in enumerate(text2)])
#     edsl_model = Model(model)
#     res = question.by(agent).by(edsl_model).run()
#     # ans = res.select('answer.*').to_list()[0]
#     return res


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