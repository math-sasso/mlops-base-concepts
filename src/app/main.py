import os
from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import pickle

colunas = ['tamanho','ano','garagem']
# model_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..','models','modelo.sav'))
model_path = os.path.join('models','modelo.sav')
modelo = pickle.load(open(model_path,'rb'))


app = Flask(__name__)

print(os.environ.get('BASIC_AUTH_USERNAME'))#None
print(os.environ.get('BASIC_AUTH_PASSWORD'))#None
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')
# app.config['BASIC_AUTH_USERNAME'] = 'matheus'
# app.config['BASIC_AUTH_PASSWORD'] = 'alura'

basic_auth = BasicAuth(app)

@app.route('/')
def home():
    return "Minha primeira API."

@app.route('/sentimento/<frase>')
@basic_auth.required
def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(to='en')
    polaridade = tb_en.sentiment.polarity
    return "polaridade: {}".format(polaridade)

@app.route('/cotacao/', methods=['POST'])
@basic_auth.required
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])

app.run(debug=True,host='0.0.0.0')
