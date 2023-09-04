import boto3
from IPython.display import Image, display
from trp import Document
from textractprettyprinter.t_pretty_print_expense import get_string, Textract_Expense_Pretty_Print, Pretty_Print_Table_Format
import io
from PIL import Image, ImageDraw
import pandas as pd
from textractcaller.t_call import call_textract, Textract_Features
from textractprettyprinter.t_pretty_print import convert_table_to_list
import os
import boto3
from textractprettyprinter.t_pretty_print_expense import get_string, Textract_Expense_Pretty_Print, Pretty_Print_Table_Format
from dotenv import load_dotenv
from dotenv import dotenv_values
load_dotenv()
aws_access_key_id = os.getenv("aws_access_key_id")
aws_secret_access_key = os.getenv("aws_secret_access_key")
print("aws_access_key_id",aws_access_key_id)
print("aws_secret_access_key",aws_secret_access_key)
def receipt_ananlysis(image_path):
  with open(image_path, 'rb') as document:
    imageBytes = bytearray(document.read())
  client =boto3.client('textract',aws_access_key_id =aws_access_key_id,aws_secret_access_key=aws_secret_access_key,region_name="us-west-2")
  response = client.analyze_expense(Document={'Bytes': imageBytes})
  # pretty_printed_string = get_string(textract_json=response, output_type=[Textract_Expense_Pretty_Print.SUMMARY, Textract_Expense_Pretty_Print.LINEITEMGROUPS], table_format=Pretty_Print_Table_Format.fancy_grid)
  return response