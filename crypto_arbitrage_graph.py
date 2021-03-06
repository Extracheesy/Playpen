# load required packages (must be previously installed)
import json
import requests
import networkx as nx
from itertools import combinations
import pandas as pd
import datetime
import time

pd.options.display.max_rows = 1000

# define the lists of ids and abbreviations
# make sure the abbreviations are in the same order as the corresponding cryptos in the id_list
id_list = ['bitcoin-cash', 'ethereum', 'bitcoin', 'litecoin', 'eos']
vs_currencies_list = ['bch', 'eth', 'btc', 'ltc', 'eos']




def get_coingecko_data(crypto_ids, crypto_abbreviations):
    # define the base url and another piece that will be added later
    base = 'https://api.coingecko.com/api/v3/simple/price?ids='
    vs_currencies_string = '&vs_currencies='

    # determine the number of ids in the list. This will help us know how many commas to add to the url
    id_remaining = len(crypto_ids)

    # loop through each coin and append its id to the url. Decrease id_remaining each iteration, add a comma to the url if it is needed
    for coin in crypto_ids:
        base += coin
        id_remaining -= 1
        if id_remaining > 0:
            base += ','

    # add the second predetermine piece to the url
    base += vs_currencies_string

    # determine the number of abbreviations in the id file. This will help us know how many commas to add to the url
    vs_remaining = len(crypto_abbreviations)

    # loop through each coin and append its abbreviation to the url. Decrease vs_remaining each iteration, add a comma to the url if it is needed
    for vs in crypto_abbreviations:
        base += vs
        vs_remaining -= 1
        if vs_remaining > 0:
            base += ','

    # print url to verify it looks correct
    # print(base)

    # request the constructed url to get the data
    request = requests.get(base)
    results_dict = json.loads(request.text)
    # print(results_dict['bitcoin-cash'])

    return (results_dict)


def define_edges(crypto_ids, crypto_abbreviations, crypto_data):
    # initialize the graph and an empty list for edges
    graph = nx.DiGraph()
    edges = []

    # define all edges and add to graph
    for coin1 in crypto_ids:
        for coin2 in crypto_ids:
            if coin1 != coin2:
                # find the index of coin1 and coin2 in the ids list. Use those values to get the abbreviations
                coin1_index = crypto_ids.index(coin1)
                coin2_index = crypto_ids.index(coin2)

                edges.append((crypto_abbreviations[coin1_index], crypto_abbreviations[coin2_index],
                              crypto_data[coin1][crypto_abbreviations[coin2_index]]))

    # print(edges)
    graph.add_weighted_edges_from(edges)

    return (graph)


def arbitrage_scan(graph):
    arb_checks = []

    for c1, c2 in combinations(graph.nodes, 2):
        #     print('paths from ', c1, 'to ', c2)
        for path in nx.all_simple_paths(graph, c1, c2):

            path_weight1 = 1
            for i in range(len(path) - 1):
                #             print(g[path[i]][path[i+1]]['weight'])
                path_weight1 *= graph[path[i]][path[i + 1]]['weight']

            #         print(f'weight for {path} is {path_weight1}')
            path.reverse()

            path_weight2 = 1
            for i in range(len(path) - 1):
                #             print(g[path[i]][path[i+1]]['weight'])
                path_weight2 *= graph[path[i]][path[i + 1]]['weight']

                #         print(f'weight for {path} is {path_weight2}')

            factor = path_weight1 * path_weight2

            arb_checks.append((path, factor))
    #         print(f'path weights factor is: {factor}')

    arb_checks = pd.DataFrame(arb_checks, columns=['Path', 'Result'])

    arb_checks['Percent'] = (arb_checks['Result'] - 1) * 100

    return (arb_checks)

def make_df_result(time, s_0, s_1, s_2, s_3, s_4, Result, Percent):
    return (pd.DataFrame({'time': time,
                          'Symbol_0': [s_0],
                          'Symbol_1': [s_1],
                          'Symbol_2': [s_2],
                          'Symbol_3': [s_3],
                          'Symbol_4': [s_4],
                          'Result': Result,
                          'Percent': Percent,
                          }))

def make_df_empty_result(time, s_0, s_1, s_2, s_3, s_4, Result, Percent):
    return (pd.DataFrame({'time': time,
                          'Symbol_0': s_0,
                          'Symbol_1': s_1,
                          'Symbol_2': s_2,
                          'Symbol_3': s_3,
                          'Symbol_4': s_4,
                          'Result': Result,
                          'Percent': Percent,
                          }))

def concat_df(df1, df2):
    list_df = [df1, df2]
    df = pd.concat(list_df, axis=0, ignore_index=True)
    return df

def fill_lst(lst):
    while len(lst) < len(id_list):
        lst.append('')
    return lst

df_result = make_df_empty_result([], [], [], [], [], [], [], [])
add_one = 0
while(1):
    results_dict = get_coingecko_data(id_list, vs_currencies_list)
    graph = define_edges(id_list, vs_currencies_list, results_dict)
    arb_checks = arbitrage_scan(graph)
    arb_checks.sort_values(by='Result', ascending=False, inplace=True)

    for i in arb_checks.index:
        if(arb_checks.loc[i,'Percent'] >= 0.5):
            print(arb_checks.loc[[i]], "   time: ", datetime.datetime.now().time())
            lst_path = arb_checks.loc[i,'Path']
            lst_path = fill_lst(lst_path)
            df = make_df_result(datetime.datetime.now().time(),
                                lst_path[0],
                                lst_path[1],
                                lst_path[2],
                                lst_path[3],
                                lst_path[4],
                                arb_checks.loc[i,'Result'],
                                arb_checks.loc[i,'Percent']
                                )
            df_result = concat_df(df_result, df)
            add_one = add_one + 1
    time.sleep(1)
    if (add_one != 0):
        df_result.to_csv('RESULT_GRAPH.csv')
    add_one = 0