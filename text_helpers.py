from PyPDF2 import PdfReader
from edsl import  Agent, Model, Survey
from edsl.questions import QuestionFreeText, QuestionLinearScale
from InstructorEmbedding import INSTRUCTOR
from scipy.spatial.distance import cosine
from consts import embedding_model 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

class TextObj():
    def __init__(self, fp, lazy_loading = True) -> None:
        self.fp = fp
        self.text_name = fp.split('/')[-1][:-4]
        self.text = self.extract_text(self.fp)
        self.cleaned_text = self.text
        self.summary = None

        if not lazy_loading:
            print("Cleaning")
                            
            # Clean
            self.clean_question, self.clean_agent = self.llm_clean_text()
            self.cleaned_text = TextObj.run_and_get(self.clean_question, self.clean_agent, Model('gpt-4-1106-preview'))

        return
    
    def set_text(self) :
        '''
        Sets the text of a text object
        '''
        self.text = TextObj.extract_text(self.fp)
        self.cleaned_text = TextObj.run_and_get(self.clean_question, self.clean_agent, Model('gpt-4-1106-preview'))
        return self.text, self.cleaned_text
    
    def update_text(self, text):
        '''
        Updates the text of a text object
        '''
        self.text = text
        return self.text
    
    def update_cleaned_text(self, cleaned_text):
        '''
        Updates the cleaned text of a text object
        '''
        self.cleaned_text = cleaned_text
        return self.cleaned_text
    
    def update_summarize_prompts(self, new_agent, new_question):
        '''
        Updates the EDSL prompts for summarizing
        '''
        self.summ_agent = new_agent
        self.summ_question = new_question
        return
    
    def update_clean_prompts(self, new_agent, new_question):
        '''
        Updates the EDSL prompts for cleaning
        '''
        self.clean_agent = new_agent
        self.clean_question = new_question
        return
    
    def set_summary(self, summary):
        self.summary = summary
        return
    
    def llm_clean_text(self, persona_instructions = 'You are an expert in formatting text.'):
        '''
        Cleans text using an LLM model
        '''
        agent = Agent(traits={'role': 'cleaner', 'persona': persona_instructions})
        question = QuestionFreeText(question_name = f'{self.text_name}', question_text = 'Nicely format the following text. Do not change any of the details, only fix spelling or grammar mistakes and fix formatting issues. Output only the cleaned text.\n\n' + self.text)
        return question, agent
    
    def summarize(self, persona_instructions = 'You are an expert in summarizing text.'):
        agent = Agent(traits={'role': 'summarizer', 'persona': persona_instructions})
        question = QuestionFreeText(question_name = f'{self.text_name}', question_text = 'Summarize the following text.\n\n' + self.cleaned_text)
        return question, agent

    @staticmethod
    def extract_text(fp):
        '''
        Extracts text from a text object
        '''
        filename = os.fsdecode(fp)
        if filename.endswith(".pdf"):
            reader = PdfReader(fp)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'

        elif filename.endswith('.txt'):
            with open(fp, 'r') as file:
                text = file.read()

        return text

    @staticmethod 
    def run_and_get(question, agent, model):
        return question.by(agent).by(model).run().select(question.question_name).to_list()[0]

    def __str__(self) -> str:
        return self.text

class Resume(TextObj):
    def __init__(self, resume_path, lazy_loading = True) -> None:
        self.modifications = []
        super().__init__(resume_path, lazy_loading)


    def summarize(self):
        persona_instructions = 'You are an expert recruiter who has been hired to summarize resumes for hiring managers to make decisions on who to hire.'
        return super().summarize(persona_instructions = persona_instructions)
    
    def add_modification(self, modification): 
        self.modifications.append(modification)
        return
    
    def pca_for_plot(self, embeddings):
        '''
        First, perform PCA to reduce the number of dimensions
        Then, use TSNE to plot the embeddings in 2D space
        Plot the first point in red and the rest in black
        '''
        pca = PCA(n_components=50)
        pca_emb = pca.fit_transform(embeddings.reshape((-1,1)))
        X_embedded = TSNE(n_components=2).fit_transform(pca_emb)

        plt.figure()

        for i in range(X_embedded.shape[0]):
            if i == 0:
                plt.scatter(X_embedded[i, 0], X_embedded[i, 1], color = 'red', label = 'original')
            else:
                plt.scatter(X_embedded[i, 0], X_embedded[i, 1], color = 'black', label = 'modified')

        plt.legend()
        plt.show()

        return


    # def calc_embeddings(self):
    #     '''
    #     Calculate embeddings for the resume
    #     '''
    #     self.embeddings = embedding_model.encode(self.modifications)
    
    def get_similarities(self, dist_function = cosine):
        '''
        Get similarities between the original resume and the modified resumes
        '''
        mod_embeddings = embedding_model.encode(self.modifications)
        orig_embedding = embedding_model.encode(self.text)
        similarities = [dist_function(orig_embedding, emb) for emb in mod_embeddings]
        return similarities

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
    def __init__(self, job_description_path, lazy_loading = True) -> None:
        super().__init__(job_description_path, lazy_loading) 

    def summarize(self):
        persona_instructions = 'You are an expert recruiter who has been hired to summarize job descriptions for hiring managers to make decisions on who to hire.'
        return TextObj.summarize(self, persona_instructions = persona_instructions)

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
            self.texts = [Resume(fp + f) for f in os.listdir(fp) if not os.path.isdir(fp + f)]
        elif text_type == 'job_descriptions':
            self.texts = [JobDescription(fp + f) for f in os.listdir(fp) if not os.path.isdir(fp + f)]
        else:
            self.texts = [TextObj(fp + f) for f in os.listdir(fp) if not os.path.isdir(fp + f)]
        self.text_names = [t.text_name for t in self.texts]

        print(self.text_names)
        print(self.texts)

        return
    
    def get_texts(self):
        return self.texts
    
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
    def clean_eval_results(res_df):
        # I want a data frame with columns: MODEL, FILE (RESUME), SCORE, COMMENT
        # Melt down on agent and model
        res_df = res_df.melt(id_vars = ['agent.agent_name', 'model.model'], var_name = 'data', value_name = 'answer')
        # Limit to just the answers
        res_df = res_df[res_df['data'].str.contains('answer')]
        # Get the comments
        is_comment = res_df['data'].str.contains('comment')
        # Remove the answer label from the column
        res_df['data'] = [
            x[len('answer') + 1 : x.rfind('_')] 
            if x.find('comment') != -1
            else x[len('answer') + 1:]
            for x in res_df['data']
        ]
        
        # Now separate and merge
        comments = res_df[is_comment]
        scores = res_df[~is_comment]
        cleaned_res = (scores
                    .merge(comments, on = ['agent.agent_name', 'model.model', 'data'], suffixes = ('_score', '_comment'))
                    .rename(columns = {'agent.agent_name': 'agent', 'model.model':'model', 
                                        'data': 'resume', 'answer_score': 'score', 'answer_comment': 'comment'}))
        return cleaned_res
    
    def apply_and_run(self, question_list, agent, model = 'gpt-4-1106-preview'):
        '''
        Applies a function to each text object and runs it

        func: function to apply to each text object. Must have the signature func(text, text_name)
        '''
        survey = Survey(questions = question_list)
        edsl_model = Model(model)
        res = survey.by(agent).by(edsl_model).run()

        return TextPool.clean_survey(res)

    def summarize_all(self, model = 'gpt-4-1106-preview'):
        '''
        Summarizes each text object
        '''

        # Set up our questions as a survey
        question_list = [t.summarize()[0] for t in self.texts]
        agent = self.texts[0].summarize()[1]
        cleaned_res = self.apply_and_run(question_list, agent, model)

        # We ran them, so might as well apply them to the actual objects themselves.
        for q, a in cleaned_res.iterrows():
            self.texts[self.text_names.index(a['question_name'])].set_summary(a['answer'])

        return cleaned_res
    
    def clean_all(self, model = 'gpt-4-1106-preview'):
        '''
        Cleans each text object
        '''
        # Set up our questions as a survey
        question_list = [t.llm_clean_text()[0] for t in self.texts]
        agent = self.texts[0].llm_clean_text()[1]
        cleaned_res = self.apply_and_run(question_list, agent, model)

        # We ran them, so might as well apply them to the actual objects themselves.
        for q, a in cleaned_res.iterrows():
            self.texts[self.text_names.index(a['question_name'])].update_cleaned_text(a['answer'])

        return cleaned_res

    def evaluation(self, eval_options = {}):
        '''
        Evaluates a pool of text objects against a reference text object
        '''
        agent_instructions = eval_options.get('agent_instructions', f'You are an expert in evaluating {self.text_type}.')
        models = eval_options.get('models', Model('gpt-4-1106-preview'))
        reference = eval_options.get('reference', None)

        if reference is not None:
            agent_instructions += f'\n\n {reference.cleaned_text}'

        agent = Agent(traits={'role': 'evaluator', 'persona': agent_instructions})
        question_list = [QuestionLinearScale(question_name = x.text_name,
                                             question_text = eval_options.get('question', 'Evaluate the following text on a scale of 1 to 10') + "\n\n" + x.cleaned_text,
                                             question_options = eval_options.get('options', list(range(1, 11))) ) for x in self.texts]
        survey = Survey(questions = question_list)
        res = survey.by(agent).by(models).run()
        return TextPool.clean_eval_results(res.to_pandas())
    

# Example usage
# resume = Resume('resumes/business_resume.pdf')
# resume = Resume('resumes/engineering_resume.pdf', lazy_loading = False)
# # print("no cleaning")
# text, cleaned = resume.text, resume.cleaned_text
# # print(resume.extract_text(False))
# # print("with cleaning")
# # print(resume.extract_text(True))
# with open('resumes/engineering_resume_cleaned.txt', 'w') as f:
#     f.write(cleaned)

# with open('resumes/engineering_resume_not_cleaned.txt', 'w') as f:
#     f.write(text)

# print(resume.summary)


resumes = TextPool('resumes/extracted_resumes/', 'resumes')
print(resumes)
# summaries = resumes.summarize_all()
# print(summaries)

# cleaned_texts = resumes.clean_all()
# print(cleaned_texts)


# for c in cleaned_texts.iterrows():
#     with open(f'resumes/extracted_resumes/cleaned/{c[1]["question_name"]}_cleaned.txt', 'w') as f:
#         f.write(c[1]['answer'])
    # print(c[1]['question_name'])
    # print(c[1]['answer'])
    # print('-----\n')

# for t in resumes.texts:
#     print('-----\n')
#     print(t.text_name, t.summary)
        
job_description = JobDescription('posts/software_engineer_generic.txt', lazy_loading = False)
eval_options = {
    'reference' : job_description
}
evals = resumes.evaluation(eval_options)
print(evals)