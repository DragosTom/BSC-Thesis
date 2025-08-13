This repository contains my web application for my BSc Thesis: "Automatic Code Summarization Using Large Language Models". It consists of a python FastAPI and a simple frontend created
using basic HTML and CSS.

The application can not be ran without adding the checkpoints to the models in the models directory (Only deepseek 1.3B instruct and CodeT5 Base 220M can be added).
After setting up the project and adding the models in the models folder (modify the path from load_models.py), don't forget to install the libraries from requirements.txt and then run the application in the terminal using: python -m uvicorn main:app --reload
This will start the application which can be found in the browser at: http://127.0.0.1:8000

The checkpoints can be found here:
https://huggingface.co/dragostom24/CodeT5_FT
https://huggingface.co/dragostom24/Deepseek_FT

University of Groningen - Dragos Tomoiaga S5159202
