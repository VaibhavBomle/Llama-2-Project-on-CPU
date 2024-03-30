# Llama2-on-Local-CPU-Machine

# How to run?

### Steps 1:

clone the repository


### Step 2:

create a virtual environment

# commands :

conda create -n cpullama python=3.8

conda activate cpullama

pip install -r requirements.txt

Note :
You need to upload (in app.py   --> DirectoryLoader  --> path ) any PDF to get result .



# Flow
# extract Text ---> chunks --> embadding model --> vector embadding --> knowledge base sementic index --> vector DB 

# user query ---> knowldege base  

# knowldege base  --> retune Ranked result ---> passing to LLM --> return Result
