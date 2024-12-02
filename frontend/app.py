from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)


# Example Python function to process input and calculate a result
def process_user_input(user_input):
    
    # Replace this logic with your custom calculations
    result = f"Your input reversed is: {user_input[::-1]}"

    # load the NER pipeline
    ner_pipeline = pipeline("ner", grouped_entities = True)

    # intput text
    text = user_input

    # perform NER
    entities = ner_pipeline(text)

    # get the extracted entities

    entity_list = []

    # Process the extracted entities
    for entity in entities:
        # Concatenate entity and score into a string and append to the list
        entity_info = f"{entity['word']} ({entity['entity_group']}, {entity['score']:.2f})"
        entity_list.append(entity_info)
    
    return entity_list

@app.route('/')
def index():
    return render_template('index.html')  # Serve the frontend page

@app.route('/process', methods=['POST'])
def process():
    # Get user input from the form
    user_input = request.form['user_input']
    
    # Call the Python function to process the input
    calculated_result = process_user_input(user_input)
    
    # Pass the result back to the front page
    return render_template('index.html', result=calculated_result)

if __name__ == '__main__':
    app.run(debug=True)
