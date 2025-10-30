from flask import Flask, render_template, request

app = Flask(__name__)

# Home page redirects to chat
@app.route("/")
def home():
    return render_template("chat.html")

# Handle question submission
@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "").strip()
    
    if not question:
        return render_template("chat.html", answer="Please enter a question.", question="")
    
    # Backend logic (for now, just echo; later call your RAG agent)
    answer = f"You asked: {question}"  # Replace this with run_agent_query(question)
    
    return render_template("chat.html", question=question, answer=answer)


if __name__ == "__main__":
    app.run(debug=True)
