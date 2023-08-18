from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def generate_plot(run_rate, current_score, current_over):
    model = LinearRegression()
    x_train = [[i] for i in range(1, 20)]
    y_train = [run_rate * i for i in range(1, 20)]
    model.fit(x_train, y_train)

    x_test = [[20 - current_over], [15 - current_over], [10 - current_over]]
    y_pred = model.predict(x_test)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_train, y_train, color='blue', label='Training Data')
    plt.plot(x_train, model.predict(x_train), color='red', label='Regression Line')
    plt.scatter(x_test, y_pred, color='green', label='Estimated Scores')
    plt.xlabel('Overs')
    plt.ylabel('Score')
    plt.title('Cricket Score Estimation')
    plt.legend()
    
    image_stream = io.BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    image_base64 = base64.b64encode(image_stream.read()).decode('utf-8')
    
    plt.close()
    
    return image_base64

@app.route("/", methods=["GET", "POST"])
def index():
    estimated_20 = None
    estimated_15 = None
    estimated_10 = None
    plot = None

    if request.method == "POST":
        run_rate = float(request.form["run_rate"])
        current_score = int(request.form["current_score"])
        current_over = int(request.form["current_over"])

        model = LinearRegression()
        x_train = [[i] for i in range(1, 20)]
        y_train = [run_rate * i for i in range(1, 20)]
        model.fit(x_train, y_train)

        x_test = [[20 - current_over], [15 - current_over], [10 - current_over]]
        y_pred = model.predict(x_test)

        estimated_20 = round(current_score + y_pred[0], 2)
        estimated_15 = round(current_score + y_pred[1], 2)
        estimated_10 = round(current_score + y_pred[2], 2)

        plot = generate_plot(run_rate, current_score, current_over)

    return render_template("index.html", estimated_20=estimated_20, estimated_15=estimated_15, estimated_10=estimated_10, plot=plot)

if __name__ == "__main__":
    app.run(debug=True)
