import tensorflow as tf
tf.enable_eager_execution()

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

import numpy as np
import tensorflow_hub as hub
import nltk
import re
from nltk.corpus import stopwords



pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/4" 
embed = hub.load(module_url)


def get_features(texts1,texts2):
    if type(texts1) is str:
        texts1 = [texts1]
    if type(texts2) is str:
        texts2 = [texts2]
    text_data1 = embed(texts1)['outputs'].numpy()
    text_data2 = embed(texts2)['outputs'].numpy()
    
    return text_data1,text_data2

def cosine_similarity(v1, v2):
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if (not mag1) or (not mag2):
        return 0
    return np.dot(np.squeeze(np.asarray(v1)), np.squeeze(np.asarray(v2))) / (mag1 * mag2)


def test_similarity(text1, text2):
    vec1,vec2 = get_features(text1,text2)
    return cosine_similarity(vec1, vec2)



def convert_to_text(filename_master,filename_student):
    """
    This function will handle the core OCR processing of images.
    """
    text_master = pytesseract.image_to_string(Image.open(filename_master))  # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    text_student = pytesseract.image_to_string(Image.open(filename_student))
    return text_master,text_student

def evaluate_student(filename_master,filename_student):
    print('Text extraction started')
    text_master,text_student = convert_to_text(filename_master,filename_student)
    print('Answer extraction started')
    master_ans = extract_ans(text_master)
    student_ans = extract_ans(text_student)
    print('evaluation started')
    result = {}
    for i in master_ans:
        print(i)
        result[i] = test_similarity(master_ans[i],student_ans[i])
    return result
    
    
    
def extract_ans(text_data):
    mas_data = text_data.split('\n')

    i = 0
    de_data = []
    de_temp = []
    while(i < len(mas_data)):
        if(mas_data[i] != ''):
            if(mas_data[i].strip()[0].isdigit() and mas_data[i].strip()[-1] == '?'):
                if(de_temp != []):
                    temp = ' '.join(de_temp)
                    de_data.append(temp)
                    de_temp = []
                temp = mas_data[i]
                de_data.append(temp)
            if((mas_data[i].strip()[0].isdigit() and mas_data[i].strip()[-1] == '?') == False):
                de_temp.append(mas_data[i])
        if(i+1 == len(mas_data)):
            temp = ' '.join(de_temp)
            de_data.append(temp)
        i = i+1

    text_mas = {}
    i = 0
    while(i < len(de_data)):
        text_mas[de_data[i].strip()[0]] = de_data[i+1][7:]
        i = i+2
    
    return text_mas
    
#master = 'C:/Users/Bharathraj C L/Projects/upload_file_python-master/act.jpg'
#student = 'C:/Users/Bharathraj C L/Projects/upload_file_python-master/test1.jpg'

#result = evaluate_student(master, student)
