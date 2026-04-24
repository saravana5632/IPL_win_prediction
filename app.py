from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

pipe = pickle.load(open('pipe.pkl', 'rb'))

teams = ['Sunrisers Hyderabad','Mumbai Indians','Royal Challengers Bangalore',
         'Kolkata Knight Riders','Kings XI Punjab','Chennai Super Kings',
         'Rajasthan Royals','Delhi Capitals']

cities = ['Hyderabad','Bangalore','Mumbai','Kolkata','Delhi','Chennai']

@app.route('/')
def home():
    return render_template('index.html', teams=teams, cities=cities)

@app.route('/predict', methods=['POST'])
def predict():
    batting_team = request.form['batting_team']
    bowling_team = request.form['bowling_team']
    city = request.form['city']
    target = float(request.form['target'])
    score = float(request.form['score'])
    overs = float(request.form['overs'])
    wickets = int(request.form['wickets'])

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

    result = pipe.predict_proba(input_df)
    win = round(result[0][1]*100)
    loss = round(result[0][0]*100)

    return render_template('index.html',
                           teams=teams,
                           cities=cities,
                           result=f"{batting_team}: {win}% | {bowling_team}: {loss}%")

if __name__ == "__main__":
    app.run(debug=True)
