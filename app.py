from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
pipe = pickle.load(open('pipe.pkl', 'rb'))

@app.route('/')
def home():
    return "IPL Win Prediction API Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract inputs
    batting_team = data['batting_team']
    bowling_team = data['bowling_team']
    city = data['city']
    target = float(data['target'])
    score = float(data['score'])
    overs = float(data['overs'])
    wickets = int(data['wickets'])

    # Feature engineering
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets

    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    # Create dataframe
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Prediction
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    return jsonify({
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "win_probability": round(win * 100, 2),
        "loss_probability": round(loss * 100, 2)
    })

if __name__ == '__main__':
    app.run()
