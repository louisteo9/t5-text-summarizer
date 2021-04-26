# T5 Text Summarizer
## Table of Contents
1. [Introduction]()
2. [File Descriptions]()
3. [Installation]()
4. [Instructions]()
5. [Acknowledgements]()
6. [Screenshots]()

## Introduction
In this project, we will use Google's state-of-the-art T5 model to create a human-like text summarizer.

I will also share my text summarizer pipelines where I combine both extractive and abstractive methods to generate meaningful summaries for PDF documents of any length.

The medium blog post that accompanying this project can be found [here]().

## File Descriptions
### Folder: data
**Folder: output** store generated summary in text file.<br/>
**Folder: pdf** store PDF files.<br/>
**Data Pipeline Preparation.ipynb** Jupyter Notebook used to prepare data pipeline.<br/>
**extract_data.py** data pipeline used to extract text from PDF files and store data in SQLite database.<br/>
**Text.db** text data from PDF files stored in SQLite database.<br/>

### Folder: models
**Model Pipeline Preparation.ipynb** Jupyter Notebook  used to prepare summarization pipeline.<br/>
**summarization.py** summarization pipeline used to load text data, summarize text and store both the summary into SQLite database and output the summary to a text file.<br/>

### Folder: screenshots
Screenshots of the text summarizer pipelines run on Terminal.

## Installation
Four major libraries that you will need to install beyond the Anaconda distribution of Python are: **torch**, **transformers**, **bert-extractive-summarizer** and **pdfminer.six**

```pip install -r requirements.txt``` will get your environment ready to run all the codes for this project.

## Instructions
Run the following commands in the project's root directory to extract text data from pdf files, store in database and perform summarization to the text.

 - To run data pipeline that extracts data and stores in database<br/>
 ```python data/extract_data.py data/pdf data/Text.db```<br/>
 - To run summarization pipeline that generates summary (using T5 model only), outputs the summary to a text file and saves to database.<br/>
 ```python models/summarization.py data/Text.db t5-base data/output 1.0 80 150 15```
 - To run summarization pipeline that shorten the text first, followed by T5 model summarization, outputs the summary to a text file and saves to database.<br/>
 ```python models/summarization.py data/Text.db t5-base data/output 0.5 80 150 15```

## Acknowledgements
* [KNIMO](https://knimo.com) for inspiring me to build a powerful automated text summarizer.
* [Hugging Face](https://huggingface.co) for providing a great AI community that builds the future through building, training and deploying state-of-the-art models powered by the reference open source in Natural Language Processing (NLP) space.

## Screenshots
1. Run data pipeline (extract_data.py) to extract text from pdf files and save to a SQLite database.<br/>
![image]()

2. Run summarization pipeline (summarization.py) [only T5] to summarize text data, save the summary to text file and store the summary to database.<br/>
![image]()

3. Run summarization pipeline (summarization.py) [BERT & T5] to summarize text data, save the summary to text file and store the summary to database.<br/>
![image]()

    _**Note:** Key in a ratio below 1.0 (e.g 0.5) if you wish to shorten the text with BERT extractive summarization before running it through T5 summarization. It takes longer to generate a summary this way because each text is run through two different summarizers._
