from flask import Flask, render_template, request, jsonify

from chat import get_response

from flask import Flask, render_template, request, redirect, url_for, session
from flask_socketio import SocketIO, join_room, leave_room, emit
from flask_session import Session

app = Flask(__name__)

app.config["SECRET_KEY"] = "hello-vtf61"


@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "GET":
        return render_template("chat.html", session=session)

    elif request.method == "POST":
        the_question = request.form["input_text"]

        response = get_response(the_question)

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run()