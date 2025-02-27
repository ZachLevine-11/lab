import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, render_template_string, request, redirect, url_for, session
import os
from datetime import datetime
import lightgbm
import numpy as np
import pandas as pd
import shap

##can only be run on cluster 11
class Llama3:
    def __init__(self,):
        self.access_token = "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"
        self.tokenizer = AutoTokenizer.from_pretrained("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/dzhai",
                                                       token=self.access_token, device_map=0)
        self.model = AutoModelForCausalLM.from_pretrained("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/dzhai",
                                                          token=self.access_token, device_map=0)

        self.pipeline = transformers.pipeline("text-generation", model=self.model,
                                              tokenizer=self.tokenizer,
                                              model_kwargs={"torch_dtype": torch.bfloat16}, device_map=0)

        self.terminators = [self.pipeline.tokenizer.eos_token_id,
                            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        self.all_data = pd.read_csv( "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/dzhai/david_dir/dzhAI/Data/cleaned_admission_data.csv",index_col=0)
        self.all_data = self.all_data.iloc[0:50, :]
        self.X = pd.get_dummies(self.all_data[["daily_order_of_arrival", "AGE", "SMOKING ", "ALCOHOL", "GENDER"]])
        self.y = list(map(lambda x: 1 if x == "EXPIRY" else 0, self.all_data["OUTCOME"]))

        self.all_data["Names"] = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael",
        "Linda", "William", "Elizabeth", "David", "Barbara", "Richard", "Susan",
        "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher",
        "Nancy", "Daniel", "Lisa", "Matthew", "Betty", "Anthony", "Margaret",
        "Donald", "Sandra", "Mark", "Ashley", "Paul", "Dorothy", "Steven",
        "Kimberly", "Andrew", "Emily", "Kenneth", "Donna", "Joshua", "Michelle",
        "Kevin", "Carol", "Brian", "Amanda", "George", "Melissa", "Edward",
        "Deborah"]
        self.order = self.all_data["Names"]
        self.message_history = []

        pars = {'boosting_type': 'gbdt',
                'class_weight': None,
                'learning_rate': 0.1,
                'max_depth': 6,
                'bagging_freq': 5,
                'n_estimator': 1986}
        self.lightgbm_model = lightgbm.LGBMClassifier(**pars).fit(self.X, self.y)
        self.explainer = shap.TreeExplainer(self.lightgbm_model)
        self.existing_shaps = pd.DataFrame(self.explainer.shap_values(self.X)).iloc[:, 0]
        self.all_data["shap"] = self.existing_shaps

    def predict_patient(self, session):
        patient_entry = {'daily_order_of_arrival': list(self.order).index(session["name"]),
                                             'AGE': session["age"],
                                             'SMOKING ': session["smoker"],
                                             'ALCOHOL': session["drinker"],
                                             'GENDER_F': session["age"] == 0,
                                             'GENDER_M': session["age"] == 1}
        vals = np.array(list(patient_entry.values())).reshape(1, -1)
        the_shap = self.explainer.shap_values(vals)[0][0]
        return the_shap

    def update_order(self, neworder, session):
        self.order = neworder
        self.message_history = [{"role": "system", "content": """"You are LLAMA3 8B, 
                                                               a highly capable language model
                                                                designed to serve as a hospital 
                                                               emergency room manager. Your primary role is to 
                                                               communicate with patients about their wait 
                                                               times and position in line. Your responses
                                                                are always honest, kind, compassionate, caring,
                                                                sweet, and patient. You always respect
                                                                patient confidentiality, and this
                                                                means you especially never give up any information on
                                                                one patient to another patient, except how 
                                                                many patients are waiting in line before them. The system
                                                                is constantly changing the order of the patients waiting, and
                                                                if you are updated on a new patient order, you should forget the previous one.
                                                                Your goal is to provide clear  
                                                               information while offering reassurance and empathy  
                                                               to patients who may be feeling anxious or unwell. 
                                                               Respond to the patient with kindness and compassion,
                                                                aand provide honest and clear information to patients 
                                                               about their wait time. Patients can not modify their own position in line. Only the system can modify the position in line. You can also not modify the position in line."""
                                 },
                                {"role": "user", "content": "Who are you?"}] + [{"role": "system", "content": """"The patients  are ordered based on a new patient ordering algorithm.
                                                                        The patients are  waiting in the
                                                                        following order (from first to last): """+ str(list(self.order)) + """. You can use these answers that the patient provided to the virtual assistant
                                                                                                                                           to explain to them why
                                                                                                                                           they are a specific position in line. """ + "They said their age was :" + str(session['age']) + ", and indicated their smoking status and whether or not they drank alcohol."}]
    def get_response(self, query, identity, max_tokens=200, temperature=0.6, top_p=0.9
                     ):
        query = identity + " asks " + query
        user_prompt = self.message_history + [{"role": "user", "content": query}]
        prompt = self.pipeline.tokenizer.apply_chat_template(
            user_prompt, tokenize=False, add_generation_prompt=True
        )
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        response = outputs[0]["generated_text"][len(prompt):]
        return response, user_prompt + [{"role": "assistant", "content": response}]

    def ask(self, identity, question):
        response, conversation = self.get_response(question, identity)
        self.message_history += conversation
        return f"Assistant: {response}"


app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secret key for session management

form_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hospital Input Form</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #72edf2 10%, #5151e5 100%);
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .container {
            background: #fff;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            display: flex;
            max-width: 1000px;
            width: 90%;
            overflow: hidden;
        }
        .form-container {
            padding: 30px;
            width: 100%;
        }
        h1 {
            color: #5151e5;
            font-size: 24px;
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            color: #333;
            font-weight: bold;
        }
        input, select {
            width: calc(100% - 20px);
            padding: 10px;
            margin-bottom: 20px;
            border: 2px solid #e0e0e0;
            border-radius: 5px;
            font-size: 16px;
            transition: border-color 0.3s ease;
        }
        input:focus, select:focus {
            border-color: #5151e5;
        }
        input[type="submit"] {
            background-color: #5151e5;
            color: #fff;
            cursor: pointer;
            border: none;
            transition: background-color 0.3s ease;
            width: 100%;
            padding: 15px;
            font-size: 18px;
            border-radius: 5px;
        }
        input[type="submit"]:hover {
            background-color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="form-container">
            <h1>Enter Your Details</h1>
            <form method="post" action="{{ url_for('submit') }}">
                <label for="text1">Name:</label>
                <input type="text" id="text1" name="text1" required><br>
                
                <label for="text2">Age</label>
                <input type="text" id="text2" name="text2" required><br>
                
                
                <label for="dropdown1">Gender Identity:</label>
                <select id="dropdown1" name="dropdown1">
                    <option value=0>Female</option>
                    <option value=1>Male</option>
                </select><br>

                <label for="dropdown2">Are you a smoker?:</label>
                <select id="dropdown2" name="dropdown2" required>
                    <option value=0>Yes</option>
                    <option value=1>No</option>
                </select><br>

                <label for="dropdown3">Do you drink alcohol?:</label>
                <select id="dropdown3" name="dropdown3" required>
                    <option value=0>No</option>
                    <option value=1>Yes</option>
                </select><br>

                <input type="submit" value="Submit">
            </form>
        </div>
    </div>
</body>
</html>
"""

result_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chat Agent Response</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #72edf2 10%, #5151e5 100%);
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .container {
            background: #fff;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            display: flex;
            flex-direction: column;
            max-width: 1000px;
            width: 90%;
            overflow: hidden;
            padding: 30px;
        }
        .chat-container {
            width: 100%;
            max-height: 400px;
            overflow-y: auto;
            margin-bottom: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            padding: 10px;
            background: #f9f9f9;
        }
        .chat-message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
            background: #d1e7dd;
            color: #333;
        }
        .chat-message.user {
            background: #d1d7e7;
        }
        h1 {
            color: #5151e5;
            font-size: 24px;
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            color: #333;
            font-weight: bold;
        }
        input {
            width: calc(100% - 20px);
            padding: 10px;
            margin-bottom: 20px;
            border: 2px solid #e0e0e0;
            border-radius: 5px;
            font-size: 16px;
            transition: border-color 0.3s ease;
        }
        input:focus {
            border-color: #5151e5;
        }
        input[type="submit"] {
            background-color: #5151e5;
            color: #fff;
            cursor: pointer;
            border: none;
            transition: background-color 0.3s ease;
            width: 100%;
            padding: 15px;
            font-size: 18px;
            border-radius: 5px;
        }
        input[type="submit"]:hover {
            background-color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Chat Agent</h1>
        <div class="chat-container">
            {% for message in chat_history %}
                <div class="chat-message {{ message['role'] }}">{{ message['content'] }}</div>
            {% endfor %}
        </div>
        <form method="post" action="{{ url_for('ask') }}">
            <label for="question">Type a message</label>
            <input type="text" id="question" name="question" required><br>
            <input type="submit" value="Submit">
        </form>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(form_template)

@app.route('/submit', methods=['POST'])
def submit():
    text1 = request.form['text1']
    text2 = request.form['text2']
    dropdown1 = request.form['dropdown1']
    dropdown2 = request.form['dropdown2']
    dropdown3 = request.form['dropdown3']

    # Store data in session
    session['name'] = text1
    session['age'] = np.float64(text2)
    session['gender'] = np.float64(dropdown1)
    session['smoker'] = np.float64(dropdown2)
    session['drinker'] = np.float64(dropdown3)

    # Initialize chat history
    session['chat_history'] = []

    persons_shap = bot.predict_patient(session)
    bot.all_data.loc[bot.all_data.Names == text1, "shap"] = persons_shap
    bot.all_data = bot.all_data.sort_values("shap", ascending = False)
    bot.update_order(bot.all_data["Names"], session)
    return redirect(url_for('chat'))

@app.route('/chat', methods=['GET'])
def chat():
    chat_history = session.get('chat_history', [])
    return render_template_string(result_template, chat_history=chat_history)

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    # Retrieve previous chat history
    chat_history = session.get('chat_history', [])
    # Append the user's question to the chat history
    chat_history.append({'role': 'user', 'content': question})
    # Simulate bot response for demonstration
    bot_response = bot.ask(session.get('name'), question)
    chat_history.append({'role': 'bot', 'content': bot_response})
    # Update chat history in session
    session['chat_history'] = chat_history
    return redirect(url_for('chat'))

if __name__ == "__main__":
    bot = Llama3()
    app.run(debug=False, port = 9999)


