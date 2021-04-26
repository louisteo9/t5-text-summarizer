import sys
import pandas as pd
import torch
import time

from sqlalchemy import create_engine
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from summarizer import Summarizer


# functions for generating summary

def ext_sum(text, ratio=0.8):
    """
    Generate extractive summary using BERT model

    INPUT:
    text - str. Input text
    ratio - float. Enter a ratio between 0.1 - 1.0 [default = 0.8]
            (ratio = summary length / original text length)

    OUTPUT:
    summary - str. Generated summary
    """
    bert_model = Summarizer()
    summary = bert_model(text, ratio=ratio)

    return summary

def abs_sum(text, model, tokenizer, min_length=80,
                     max_length=150, length_penalty=15,
                     num_beams=2):
    """
    Generate abstractive summary using T5 model

    INPUT:
    text - str. Input text
    model - model name
    tokenizer - model tokenizer
    min_length - int. The min length of the sequence to be generated
                      [default = 80]
    max_length - int. The max length of the sequence to be generated
                      [default = 150]
    length_penalty - float. Set to values < 1.0 in order to encourage the model
                     to generate shorter sequences, to a value > 1.0 in order to
                     encourage the model to produce longer sequences.
                     [default = 15]
    num_beams - int. Number of beams for beam search. 1 means no beam search
                     [default = 2]

    OUTPUT:
    summary - str. Generated summary
    """
    tokens_input = tokenizer.encode("summarize: "+text, return_tensors='pt',
                                    # model tokens max input length
                                    max_length=tokenizer.model_max_length,
                                    truncation=True)

    summary_ids = model.generate(tokens_input,
                                min_length=min_length,
                                max_length=max_length,
                                length_penalty=length_penalty,
                                num_beams=num_beams)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

def generate_summary(text, model, tokenizer, ext_ratio=1.0, min_length=80,
                     max_length=150, length_penalty=15,
                     num_beams=2):
    """
    Generate summary for using extractive & abstractive methods

    INPUT:
    text - str. Input text
    model - model name
    tokenizer - model tokenizer
    ext_ratio - float. Enter a ratio between 0.1 - 1.0 [default = 1.0]
                (ratio = summary length / original text length)
                1.0 means no extractive summarization is performed before
                abstractive summarization
    min_length - int. The min length of the sequence to be generated
                 [default = 80]
    max_length - int. The max length of the sequence to be generated
                 [default = 150]
    length_penalty - float. Set to values < 1.0 in order to encourage the model
                     to generate shorter sequences, to a value > 1.0 in order to
                     encourage the model to produce longer sequences.
                     [default = 15]
    num_beams - int. Number of beams for beam search. 1 means no beam search
                     [default = 2]

    OUTPUT:
    summary - str. Generated summary
    """
    if ext_ratio == 1.0:
        summary = abs_sum(text, model, tokenizer, min_length,
                       max_length, length_penalty, num_beams)
    elif ext_ratio < 1.0:
        text = ext_sum(text, ratio = ext_ratio)
        summary = abs_sum(text, model, tokenizer, min_length,
                       max_length, length_penalty, num_beams)
    else:
        print('Error! Please enter ext_ratio betwen 0.1 and 1.0')

    return summary

# function to generate summary and save output to list and text file
# please be patience, as this loop will take some time to run
def gen_sum_save_monitor(df, model, tokenizer, output_folder, ext_ratio=1.0,
                         min_length=80, max_length=150, length_penalty=15,
                         num_beams=2):
    """
    Monitor progress while generating summary & save output to list & text file

    INPUT:
    df - DataFrama. Data loaded from database
    model - model name
    tokenizer - model tokenizer
    output_folder - str. Folder name to save the generated output in text file
    ext_ratio - float. Enter a ratio between 0.1 - 1.0 [default = 1.0]
                (ratio = summary length / original text length)
                1.0 means no extractive summarization is performed before
                abstractive summarization
    min_length - int. The min length of the sequence to be generated
                 [default = 80]
    max_length - int. The max length of the sequence to be generated
                 [default = 150]
    length_penalty - float. Set to values < 1.0 in order to encourage the model
                     to generate shorter sequences, to a value > 1.0 in order to
                     encourage the model to produce longer sequences.
                     [default = 15]
    num_beams - int. Number of beams for beam search. 1 means no beam search
                [default = 2]

    OUTPUT:
    summaries - list. Generated summary appended to a list
    """
    # create an empty list
    summaries = []
    # loop through each raw_text row in dataset, generate summary,
    # append summary to summaries list
    for i in range(len(df)):
        file_path = df.file_path[i]
        raw_text = df.raw_text[i]

        start = time.time()
        summary = generate_summary(raw_text, model, tokenizer,
                                   ext_ratio, min_length, max_length,
                                   length_penalty, num_beams)

        file_name = file_path[8:][:-4]+'_summary.txt'

        with open(output_folder + "/" + file_name, 'w')as text_file:
            text_file.write(summary)


        summaries.append(summary)
        end = time.time()
        print(" Summarized '{}'[time: {:.2f}s]".format(file_path,
                                                       end-start))

    return summaries

def main():
    if len(sys.argv) == 8:
        database_filepath, model_name, output_folder, ext_ratio, min_length, \
        max_length, length_penalty = sys.argv[1:]

        print('\nLoading data...\n DATABASE: {}'.format(database_filepath))
        engine = create_engine('sqlite:///'+database_filepath)
        df = pd.read_sql_table('Text_table', engine)

        print('\nLoading pre-train model and tokenizer...'\
              '\n MODEL : {}'.format(model_name))
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print(' {} model has been loaded'.format(model_name))
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(' {} tokenizer has been loaded'.format(model_name))

        print('\nGenerating summary...')
        print('Please be patience, this will take some time to run')
        summaries = gen_sum_save_monitor(df, model, tokenizer, output_folder,
                                    float(ext_ratio), int(min_length),
                                    int(max_length), float(length_penalty))

        # create a new 'summary' column and append to df
        df['summary'] = summaries

        print('\nSaving summary...\n DATABASE: {}'.format(database_filepath))
        engine = create_engine('sqlite:///'+database_filepath)
        df.to_sql('Text_table', engine, if_exists = 'replace', index=False)

        print('\nSummary saved to database')
    else:
        print('Please provide: '\
            '\n 1) first argument: filepath to text database'\
            '\n 2) second argument: model name'\
            '\n 3) third argument: filepath to output folder'\
            '\n 4) forth argument: ext_ratio'\
            '\n 5) fifth argument: min_length'\
            '\n 6) sixth argument: max_length'\
            '\n 7) seventh argument: length_penalty'
            ' \n\n Example: python models/summarization.py data/Text.db t5-base '\
            'data/output 1.0 80 150 15')

if __name__ == '__main__':
    main()
