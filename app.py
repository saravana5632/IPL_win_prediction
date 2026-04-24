from flask import Flask, request, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)


model_path = os.path.join(os.path.dirname(__file__), 'pipe.pkl')
pipe = pickle.load(open(model_path, 'rb'))

teams = ['Sunrisers Hyderabad','Mumbai Indians','Royal Challengers Bangalore',
         'Kolkata Knight Riders','Kings XI Punjab','Chennai Super Kings',
         'Rajasthan Royals','Delhi Capitals']

cities = ['Hyderabad','Bangalore','Mumbai','Kolkata','Delhi','Chennai']

@app.route('/')
def home():
    return render_template('index.html', teams=teams, cities=cities)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        batting_team = request.form.get('batting_team')
        bowling_team = request.form.get('bowling_team')
        city = request.form.get('city')

        target = float(request.form.get('target', 0))
        score = float(request.form.get('score', 0))
        overs = float(request.form.get('overs', 0))
        wickets = int(request.form.get('wickets', 0))

        # Validation
        if batting_team == bowling_team:
            return render_template('index.html', teams=teams, cities=cities,
                                   result="Teams cannot be same")

        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets

        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        input_df = pd.DataFrame({
            'batting_team':[batting_team],
            'bowling_team':[bowling_team],
            'city':[city],
            'runs_left':[runs_left],
            'balls_left':[balls_left],
            'wickets':[wickets_left],
            'total_runs_x':[target],
            'crr':[crr],
            'rrr':[rrr]
        })

        print(input_df)  # DEBUG

        result = pipe.predict_proba(input_df)

        win = round(result[0][1] * 100)
        loss = round(result[0][0] * 100)

        return render_template('index.html',
                               teams=teams,
                               cities=cities,
                               result=f"{batting_team}: {win}% | {bowling_team}: {loss}%")

    except Exception as e:
        return f"ERROR: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
