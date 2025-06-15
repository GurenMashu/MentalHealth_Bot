Directory Structure ->

MentalHealth_bot
|
|-data
   |---save_load_data
   |---dataset.csv    #after running flow.py
|
|-rag
   |---populate_db.py
   |---retrieve.py
|
|-app.py
|-flow.py
|-.env
|-README.md
|-requirements.txt


Instructions to run:-
1. Setup the requirements using requirements.txt
```bash
python venv -m BIAenv
BIAenv\Scripts\activate #for windows
pip install -r requirements.txt
```
2. run flow.py to save dataset from huggingface as .csv
```bash
python flow.py
```
3. run app.py to launch the streamlit app
```bash
streamlit run app.py
```

##Important:-
Setup a .env file with your GEMINI_API
```bash
#.env
GEMINI_API = "your api key here"
```