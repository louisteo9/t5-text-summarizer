import sys
import numpy as np
import pandas as pd

from glob import glob
from pdfminer.high_level import extract_text
from sqlalchemy import create_engine

def main():

    if len(sys.argv) == 3:
        folder_filepath, database_filepath = sys.argv[1:]

        print('\nLoading filenames...')
        pdf_files = np.array(glob(folder_filepath+'/*'))

        for i in range(len(pdf_files)):
            print(" "+pdf_files[i])

        text_dict = {}

        print('\nExtracting text from pdf files...')
        for file in pdf_files:
            text = extract_text(file)
            text_dict[file] = text

        df = pd.DataFrame(list(text_dict.items()),
                        columns = ['file_path', "raw_text"])

        print('\nSaving data...\n DATABASE: {}'.format(database_filepath))
        engine = create_engine('sqlite:///'+ database_filepath)
        df.to_sql('Text_table', engine, if_exists = 'replace', index=False)

        print('\nData saved to database!')

    else:
        print('Please provide filepath of the folder containing pdf files '\
               'as first argument and filepath of the database to save the '\
               'extracted text. \n\nExample: python data/extract_data.py '\
               'data/pdf data/Text.db')

if __name__ == '__main__':
    main()
