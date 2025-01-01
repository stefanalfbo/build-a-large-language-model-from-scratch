# This script downloads the text file "the-verdict.txt" from Sebastian Raschkas GitHub repository
import urllib.request

url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

file_path = "data/the-verdict.txt"

urllib.request.urlretrieve(url, file_path)