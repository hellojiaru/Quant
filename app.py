import json
import datetime
import numpy as np
from flask import Flask, jsonify, request, render_template
from db_content import SqlServer

app = Flask(__name__)

db = SqlServer()
db_name = 'Simulated_Transaction'
avg_list = ['v1', 'v2']


@app.route('/')
def index():
    model = 'v2' if request.values.get("m") is None else request.values.get("m")
    date = request.values.get("d")
    t1 = 0.1 if request.values.get("t1") is None else request.values.get("t1")
    t2 = 0.9 if request.values.get("t2") is None else request.values.get("t2")
    p_j_str = None

    if model is not None and date is not None:
        
        if model == 'avg':
            sql = "select * from {0} where Model in ({1}) and Date='{2}'".format(db_name, ','.join([f"'{s}'" for s in avg_list]), date)
            result = db.get_sql(sql)
            if len(result) != 0:
                stock_probs = []
                for i in range(len(result)):
                    stock_probs.append(json.loads(result[i]['Stock_Prob']))
                values = []
                for i in range(len(stock_probs)):
                    values.append(np.array(list(stock_probs[i].values()))[np.newaxis, :])
                keys = stock_probs[0].keys()
                mean_values = np.mean(np.concatenate(values), axis=0)
                p_j_str = json.dumps(dict(sorted(dict(zip(keys, mean_values)).items(), key=lambda x: x[1], reverse=True)))
        else:
            sql = "select * from {0} where Model='{1}' and Date='{2}'".format(db_name, model, date)
            result = db.get_sql(sql)
            if len(result) != 0:
                p_j_str = json.dumps(dict(sorted(json.loads(result[0]['Stock_Prob']).items(), key=lambda x: x[1], reverse=True)))

    with open(r'static\position.json', 'r', encoding='utf-8') as f:
        position = json.dumps(json.load(f), ensure_ascii=False)

    return render_template('index.html', model=model, date=date, t1=t1, t2=t2, p_j_str=p_j_str, position=position)


@app.route('/update_pos', methods=['POST'])
def update_pos():
    position = json.loads(request.get_data(as_text=True))
    with open(r'static\position.json', 'w', encoding='utf-8') as f:
        b = json.dumps(position, ensure_ascii=False)
        f.write(b)
    return "success"


@app.route('/get_sotck_probability/<model>/<date>')
def get_sotck_probability(model, date):
    return_json = {}

    if model == 'avg':
        ms = ','.join([f"'{s}'" for s in avg_list])

        if 'to' in date:
            start, end = date.split('to')
            sql = "select * from {0} where Model in ({1}) and Date >= '{2}' and Date <= '{3}' ORDER BY Date".format(db_name, ms, start, end)
            result = db.get_sql(sql)

            for i in range(0, len(result), len(avg_list)):
                day_datas = result[i:i+len(avg_list)]

                stock_probs = []
                for i in range(len(day_datas)):
                    stock_probs.append(json.loads(day_datas[i]['Stock_Prob']))
                values = []
                for i in range(len(stock_probs)):
                    values.append(np.array(list(stock_probs[i].values()))[np.newaxis, :])
                keys = stock_probs[0].keys()
                mean_values = np.mean(np.concatenate(values), axis=0)

                return_json[day_datas[0]['Date'].strftime('%Y-%m-%d')] = json.dumps(dict(zip(keys, mean_values)))
        else:
            sql = "select * from {0} where Model in ({1}) and Date='{2}'".format(db_name, ms, date)
            result = db.get_sql(sql)
            stock_probs = []
            for i in range(len(result)):
                stock_probs.append(json.loads(result[i]['Stock_Prob']))
            values = []
            for i in range(len(stock_probs)):
                values.append(np.array(list(stock_probs[i].values()))[np.newaxis, :])
            keys = stock_probs[0].keys()
            mean_values = np.mean(np.concatenate(values), axis=0)
            return_json = json.dumps(dict(zip(keys, mean_values)))
    else:
        if 'to' in date:
            start, end = date.split('to')
            sql = "select * from {0} where Model='{1}' and Date >= '{2}' and Date <= '{3}' ORDER BY Date".format(db_name, model, start, end)
            result = db.get_sql(sql)
            return_json = {item['Date'].strftime('%Y-%m-%d'): item['Stock_Prob'] for item in result}
        else:
            sql = "select * from {0} where Model='{1}' and Date='{2}'".format(db_name, model, date)
            result = db.get_sql(sql)
            assert len(result) == 1
            return_json = result[0]['Stock_Prob']

    return jsonify(return_json)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='9527', debug=True)
