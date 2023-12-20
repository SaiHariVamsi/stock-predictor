from flask import Flask, render_template
import pandas as pd
import os
import atexit

app = Flask(__name__)

def cleanup():
    if os.path.exists('static/plot.png'):
        os.remove('static/plot.png')
    print('cleaned')

atexit.register(cleanup)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_prediction/<company>')
def run_prediction(company):
    script_mapping = {
        'tata_steel': 'si_tata_steel.py',
        'tcs' : 'si_tcs.py',
        'hdfc' : 'si_hdfc.py',
        'tata_power' : 'si_tata_power.py',
        'lic' : 'si_lic.py',
        'irctc' : 'si_irctc'
    }
    if company in script_mapping:
        command = f'python {script_mapping[company]}'
        print(f'Executing command: {command}')
        os.system(command)

        return render_template('result.html', company_name=company.capitalize())
    else:
        return 'Invalid company selection', 400

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
