from flask import Flask,request,render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Retrieve form data
            gender = request.form.get('gender')
            race_ethnicity = request.form.get('race/ethnicity')
            parental_level_of_education = request.form.get('parental level of education')
            lunch = request.form.get('lunch')
            test_preparation_course = request.form.get('test preparation course')
            reading_score = float(request.form.get('reading score'))
            writing_score = float(request.form.get('writing score'))

            # Create CustomData instance
            data = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=reading_score,
                writing_score=writing_score
            )

            # Convert data to DataFrame
            pred_df = data.get_data_as_dataframe()

            # Convert DataFrame to dictionary
            input_data = pred_df.iloc[0].to_dict()

            # Make prediction
            predict_pipeline = PredictPipeline()
            prediction = predict_pipeline.predict(input_data)

            # Render the result on the same page
            return render_template('home.html', results=f'Predicted Math Score: {prediction}')

        except Exception as e:
            return render_template('home.html', results=f'Error: {str(e)}')

    # Render the form if GET request
    return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)