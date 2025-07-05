# running the pyhton project

- start up the folder 
- change dir to the folder where the environment( venv ) is located
- start the environment : `source venv/bin/activate`
- navigate back to where the `app.py` is located which is : `xray-classier/backend/app.py`
- run the python app.py with streamlit : `streamlit run app.py`

after running 
the broswer automatically loads : `http://localhost:8501/ `
add a image from the imageDataset and predict 

# learning 
> train = images used to teach the model

    These are the images the model sees and learns patterns from.

> val = validation set (images the model doesn't learn from but uses for testing during training)

    These images help evaluate if the model is generalizing well, not just memorizing.

`split_data.py` was used to split that data into the destined folder since model return just one disease
 
 > test against this : 
 Cardiomegaly
 Fracture
 Normal
 Other
 Pneumonia
 Pneumothorax
 PulmonaryEdema

 model uses the train/val folder name to train and give prediction , make sure the folder you are getting from is the right one 