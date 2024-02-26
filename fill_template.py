import docx2txt
import re

def extract_from_docx(template_fp):
    '''
    Extracts text from a docx file
    '''
    # Read the docx file
    text = docx2txt.process(template_fp)
    file_base = template_fp.split('.docx')[0]
    write_file = f'{file_base}.txt'

    with open(write_file, 'w') as f:
        f.write(text)

    return text, write_file

def read_template(template_fp):
    '''
    Fills a template with the contents of a pandas dataframe
    '''
    # Load the template
    with open(template_fp, 'r') as f:
        template = f.read()

    template_split = re.split('(<[^>]*>)', template)
    
    new_template, params = [], []

    for part in template_split:
        if part.startswith('<') and part.endswith('>'):
            part = part[1:-1]
            params.append(part)
            new_template.append('{' + part + '}')
        else:
            new_template.append(part)

    print(template_split)
    
    # Fill the template
    # filled_template = template.format(**res_df.iloc[0])
    
    return new_template, params

print(read_template('outlines/basic_resume_2.txt'))
# extract_from_docx('outlines/basic_resume.docx')

