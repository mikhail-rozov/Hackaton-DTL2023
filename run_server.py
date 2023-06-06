import annoy
from flask import Flask, request, jsonify
import logging
import pandas as pd
import pickle
import re
import torch
from transformers import AutoTokenizer, AutoModel

from hidden import model_path

app = Flask(__name__)
logger = logging.getLogger('werkzeug')
logger.setLevel(logging.ERROR)


@app.route('/', methods=['GET'])
def general():
    return 'Hackaton chat model'


@app.route('/question', methods=['POST'])  # Принимает JSON вида {'id': str, 'question': str, 'new_chat': bool}
def get_answer():                          # Возвращает JSON вида {'answer': str, 'success': bool}
    global client_info

    request_json = request.get_json()
    data = {'success': False}

    text = request_json['question']
    user_id = request_json['id']
    is_new_chat = request_json['new_chat']

    # Защита от длинных запросов
    if len(text) > 1500:
        data['answer'] = 'Пожалуйста, перефразируйте более коротко.'
        data['success'] = True
        return jsonify(data)

    # В случае, если пользователь не знаком боту или поступила команда на перезапуск чата, данные о пользователе
    # приводятся к дефолтным значениям
    if is_new_chat or user_id not in client_info.keys():
        clear_user(user_id)

    idx = client_info[user_id]['idx']
    curr_req = client_info[user_id]['curr_req']

    # Ветка 3 - находимся внутри запроса, и бот предложил вывести ответственность
    if curr_req is not None:
        if text == '1':
            answer = df_2.loc[df_2['Question'] == curr_req, 'Resp'].iloc[0] + '\n\n' + next_query()

            client_info[user_id]['curr_req'] = None
            client_info[user_id]['idx'] = None

        elif text == '0':
            answer = get_req_list(idx)
            client_info[user_id]['curr_req'] = None

        else:
            answer = 'Введите 1 для ознакомления с ответственностью за невыполнение данного требования. Либо ' \
                     'введите 0 для возврата на предыдущий шаг.'

        data['success'] = True

    # Ветка 2 - находимся внутри запроса, и бот спрашивает о пункте требований
    elif text.isdigit() and idx is not None:
        if int(text) > df_1['n_answers'].iloc[idx] or text == '0':
            answer = 'Данный индекс не входит в диапазон.'

        else:
            # 47.5 µs ± 506 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
            requirement = re.findall(f'({text}\) )(.*)\n\n\n', df_1.iloc[idx]['Answer'])[0][1]

            # 241 µs ± 5.32 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
            answer = 'НПА про ' + requirement.lower()[:-1] + ':' + '\n\n' + df_2.loc[df_2['Question'] ==
                                                                                requirement, 'Answer'].iloc[0] + \
                     '\n\n' + 'Могу рассказать об ответственности за невыполнение данного требования. Для ' \
                              'этого введите 1. Либо введите 0 для возврата на предыдущий шаг.'

            client_info[user_id]['curr_req'] = requirement

        data['success'] = True

    # Ветка 1 - находимся вне запроса, и бот ждёт либо запроса на предоставление НПА по какому-то требованию, либо
    # на вывод списка ответственности
    else:
        # 391 µs ± 5.82 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
        not_about_resp = lr_model.predict([text])

        # CPU - 47.3 ms ± 1.1 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
        # GPU - 5.5 ms ± 21.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
        sentence = embed_sentence(text)

        if not_about_resp:
            # 6.14 ms ± 21.9 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
            index, distance = annoy_index.get_nns_by_vector(sentence, 1, include_distances=True)

            if distance[0] > 0.68:
                answer = no_answer()
            else:
                idx = index[0]
                client_info[user_id]['idx'] = idx
                answer = get_req_list(idx)
                data['success'] = True

        else:
            # 6.49 ms ± 47.7 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
            index, distance = annoy_index_resp.get_nns_by_vector(sentence, 1, include_distances=True)

            if distance[0] > 0.68:
                answer = no_answer()
            else:
                answer = df_2['Resp'].iloc[index[0]] + '\n\n' + next_query()
                data['success'] = True

    data['answer'] = answer

    return jsonify(data)


def clear_user(user_id: str) -> None:
    global client_info
    client_info[user_id] = {'idx': None,
                            'curr_req': None}


def embed_sentence(sentence):
    text = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').to(model.device)

    with torch.inference_mode():
        model_output = model(**text)

    token_embeddings = model_output[0]
    expanded_mask = text['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()

    return (torch.sum(token_embeddings * expanded_mask, dim=1) /
            torch.clamp(expanded_mask.sum(dim=1), min=1e-9)).squeeze()


def get_req_list(idx: int) -> str:
    return df_1['Answer'].iloc[idx] + f'Могу рассказать о нормативно-правовых актах по любому из ' \
                                      f'этих требований. Введите число от 1 до ' \
                                      f'{df_1["n_answers"].iloc[idx]} для выбора требования. Либо ' \
                                      f'введите новый запрос.'


def next_query():
    return 'Пожалуйста, введите следующий запрос.'


def no_answer():
    return 'К сожалению, у меня нет ответа на этот вопрос. Пожалуйста, постарайтесь переформулировать ' \
           'его или запишитесь на консультацию.'


if __name__ == '__main__':
    df_1 = pd.read_csv('./qa_dataset_1.csv')
    df_2 = pd.read_csv('./qa_dataset.csv')
    model_name = model_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModel.from_pretrained(model_name).to(device)
    annoy_index = annoy.AnnoyIndex(model.config.hidden_size, 'angular')
    annoy_index.load('./models/qa_annoy_model_1.ann')
    annoy_index_resp = annoy.AnnoyIndex(model.config.hidden_size, 'angular')
    annoy_index_resp.load('./models/qa_annoy_model_resps.ann')
    with open('./models/lr_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    client_info = {}

    app.run(host='0.0.0.0', port=8080)
