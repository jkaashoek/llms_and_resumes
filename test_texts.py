from text_helpers import TextObj, Resume, JobDescription, TextPool

# Example usage
# resume = Resume('resumes/business_resume.pdf')
# resume = Resume('resumes/engineering_resume.pdf', lazy_loading = True)

# Let's use the Kaggle data and not doing any sort of llm or cleaning
# kaggle_dir = 'resumes/kaggle_resumes/'
# kaggle_resumes = TextPool(kaggle_dir, 'resumes')

# Separation
# print("All kaggle separation", kaggle_resumes.calc_separation())
# kaggle_resumes.plot_texts(label_points = True, kaggle = True)

kaggle_dir = 'resumes/kaggle_small/'
kaggle_small = TextPool(kaggle_dir, 'resumes')

# Separation
print("All kaggle small separation", kaggle_small.calc_separation())
kaggle_small.plot_texts(label_points = True, kaggle = False)


# Smaller dataset has less separation, which would be bad if that weren't the case.
# We could start to look at differences at the resume section level



# with open('resumes/extracted_resumes/business_resume.txt', 'r') as f:
#     res2 = f.read()

# with open('resumes/extracted_resumes/technology_resume.txt', 'r') as f:
#     res3 = f.read()

# with open('resumes/extracted_resumes/health_related_resume.txt', 'r') as f:
#     res4 = f.read()

# resume.add_modification(res2)
# resume.add_modification(res3)
# resume.add_modification(res4)

# m_emb, o_emb = resume.calc_embeddings()
# print(resume.get_similarities()) 

# # Stac
# emb = np.vstack((m_emb, o_emb))

# print(emb.shape)

# # Plot
# resume.pca_for_plot(emb)



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


# resumes = TextPool('resumes/extracted_resumes/', 'resumes')
# print(resumes)
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
        
# job_description = JobDescription('posts/software_engineer_generic.txt', lazy_loading = False)
# eval_options = {
#     'reference' : job_description
# }
# evals = resumes.evaluation(eval_options)
# print(evals)