from flask import Flask, render_template, request
from core.agent import Agente

app = Flask(__name__)
agente = Agente()


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        arquivo = request.files["file"]
        target = request.form["target"]
        task = request.form["task"]
        formato = request.form.get("formato", "csv")
        modelo = request.form.get("modelo", "lightgbm")
        otimizar = request.form.get("otimizar") == "on"
        n_trials = int(request.form.get("n_trials", 30))

        caminho = f"temp_{arquivo.filename}"
        arquivo.save(caminho)

        config = {
            "caminho": caminho,
            "formato": formato,
            "target": target,
            "task": task,
            "modelo": modelo,
            "otimizar": otimizar,
            "n_trials": n_trials
        }

        resultado = agente.analisar_dados(config)

        return render_template(
            "index.html",
            resultado=resultado
        )

    return render_template("index.html")
