# /usr/bin/env python
# coding=utf-8
from collections import defaultdict
import json
import pandas as pd
import os


def precess():
    for file in ['train', 'test', 'dev']:
        dirname = f'/home/hjj/code/RelationExtract/Joint_Extraction/PRGC/data/baidu'
        data_new_list = []
        with open(f'{dirname}/{file}.json', "r", encoding='utf-8') as f:
            for line in f:
                data_new = {}
                data = json.loads(line)
                data_new['text'] = data['text']
                data_new['triple_list'] = []
                for ele in data['spo_list']:
                    data_new['triple_list'].append([ele['subject'], ele['predicate'], ele['object']])
                data_new_list.append(data_new)

        with open(f'{dirname}/{file}_triples.json', 'w', encoding='utf-8') as f:
            json.dump(data_new_list, f, indent=4, ensure_ascii=False)

    print('Finish')


def rel2id():
    dirname = f'../data/baidu'
    file = 'rel'
    data_new_list = []
    data = json.load(open(f'{dirname}/{file}.json', encoding='utf8'))
    
    data_word2id = {}
    for k, v in data.items():
        data_word2id[v] = int(k)

    data_new_list.append(data)
    data_new_list.append(data_word2id)

    with open(f'{dirname}/rel2id.json', 'w', encoding='utf-8') as f:
        json.dump(data_new_list, f, indent=4, ensure_ascii=False)

    print('finish')


def trans_baidu_data():
    relation = [None, None, None]
    for file in ['train', 'test', 'dev']:
        # dirname = f'/home/hjj/code/RelationExtract/DeepKE/example/re/standard/'
        dirname = os.getcwd()
        data_new_list = []
        out_df = pd.DataFrame(columns=['sentence', 'relation', 'head', 'tail'])
        with open(f'{dirname}/data/baidu/{file}.json', "r", encoding='utf-8') as f:
            for ind, line in enumerate(f):
                if ind > 10000:
                    break
                data = json.loads(line)
                for ele in data['spo_list']:
                    out_df = out_df.append({'sentence': data['text'], 'relation': ele['predicate'],
                                            'head': ele['subject'], 'tail': ele['object']}, ignore_index=True)
                    # data_new = {}
                    # data_new['sentence'] = data['text']
                    # data_new['relation'] = ele['predicate']
                    # data_new['head'] = ele['subject']
                    # data_new['tail'] = ele['object']

                    rel = [ele['subject_type'], ele['object_type'], ele['predicate']]
                    if rel not in relation:
                        relation.append(rel)

        out_df.to_csv(f'{dirname}/data/baidu/{file}.csv', index=False)

    relation.insert(0, [None, None, None])
    relations = pd.DataFrame(relation)
    relations['index'] = [i for i in range(relations.shape[0])]
    relations.columns = ['head_type', 'tail_type', 'relation', 'index']

    # relations.to_csv(f'{dirname}/data/baidu/relation.csv', index=False)

    print('完成')


if __name__ == '__main__':
    # precess()
    # rel2id()
    trans_baidu_data()