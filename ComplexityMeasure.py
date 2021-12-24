import os
import pyodbc as sql
import numpy as np
import pandas as pd
from ComplexityModels import MLP as MLP

pieces_order = 'KQRBNPkqrbnp'
ind = {pieces_order[i]: i for i in range(12)}
#print("Ind", ind)

# Convert board state FEN to 64 character string
def replace_tags_board(board_san):
    board_san = board_san.replace('2', '11')
    board_san = board_san.replace('3', '111')
    board_san = board_san.replace('4', '1111')
    board_san = board_san.replace('5', '11111')
    board_san = board_san.replace('6', '111111')
    board_san = board_san.replace('7', '1111111')
    board_san = board_san.replace('8', '11111111')
    return board_san.replace('/', '')

# Convert FEN to 13 x 64 matrix/832 value array
"""
12 rows for each distinct piece type, 1 row for additional variables like whose move and castling priviliges
Since FEN's are order starting at the 8th rank from White's point of view, logically makes more sense to order arrays a8-h1 instead of a1-h8
"""
def convert_fen(inp_str): # 1 x 832 array
    fen = inp_str.split('|')[0]
    board_state = replace_tags_board(fen.split(' ')[0])
    to_move = fen.split(' ')[1]
    castling = fen.split(' ')[2]

    position_matrix = np.zeros(shape=(13, 64), dtype=int)

    # piece positioning arrays
    for idx, char in enumerate(board_state):
        if char.isalpha():
            position_matrix[ind[char]][idx] = 1
    
    # additional variable array
    position_matrix[12][0] = 1 if to_move == 'w' else 0 # whose move
    position_matrix[12][1] = 1 if 'K' in castling else 0 # White kingside castling
    position_matrix[12][2] = 1 if 'Q' in castling else 0 # White queenside castling
    position_matrix[12][3] = 1 if 'k' in castling else 0 # Black kingside castling
    position_matrix[12][4] = 1 if 'q' in castling else 0 # Black queenside castling
    #position_matrix[12][5] = inp_str.split('|')[1] # rating
    # position_matrix[12][5] through position_matrix[12][63] still unused

    assert position_matrix.shape == (13, 64)
    position_matrix_flattened = np.concatenate(position_matrix)
    return position_matrix_flattened

def convert_array(position_array): # turn array back to FEN
    to_move = 'w' if position_array[768] == 1 else 'b'
    castling = 'K' if position_array[769] == 1 else ''
    castling = castling + 'Q' if position_array[770] == 1 else castling
    castling = castling + 'k' if position_array[771] == 1 else castling
    castling = castling + 'q' if position_array[772] == 1 else castling
    castling = '-' if castling == '' else castling

    reconstructed_fen = ''
    for i in range(64): # a8-h1
        for j in range(12): # cycle through each piece
            if position_array[i + 64 * j] == 1:
                reconstructed_fen = reconstructed_fen + pieces_order[j]
                break
            if j == 11: # only will get here if square is empty
                reconstructed_fen = reconstructed_fen + '1'
    
    reconstructed_fen = '/'.join(reconstructed_fen[i:i+8] for i in range(0, len(reconstructed_fen), 8)) # add a / every 8 characters
    reconstructed_fen = reconstructed_fen.replace('11111111', '8')
    reconstructed_fen = reconstructed_fen.replace('1111111', '7')
    reconstructed_fen = reconstructed_fen.replace('111111', '6')
    reconstructed_fen = reconstructed_fen.replace('11111', '5')
    reconstructed_fen = reconstructed_fen.replace('1111', '4')
    reconstructed_fen = reconstructed_fen.replace('111', '3')
    reconstructed_fen = reconstructed_fen.replace('11', '2')

    # currently no way to extract current move or 50-move counter, not important I say
    reconstructed_fen  = reconstructed_fen + ' ' + to_move + ' ' + castling  + ' - 0 0'
    return reconstructed_fen

def main():
    conn = sql.connect('Driver={ODBC Driver 17 for SQL Server};Server=HUNT-PC1;Database=ChessAnalysis;Trusted_Connection=yes;')    
    class_train_qry = '''
SELECT
Input_String,
Best_Move
FROM vwPositionComplexity
WHERE GameID <= 20000
'''
    class_test_qry = '''
SELECT
Input_String,
Best_Move
FROM vwPositionComplexity
WHERE GameID > 20000
'''

    acpl_train_qry = '''
SELECT
Input_String,
CP_Loss
FROM vwPositionComplexity
WHERE GameID <= 20000
AND CP_Loss > 0
'''
    acpl_test_qry = '''
SELECT
Input_String,
CP_Loss
FROM vwPositionComplexity
WHERE GameID > 20000
AND CP_Loss > 0
'''
    
    weight_path = r'C:\Users\eehunt\Repository\PositionComplexity\weights'

    lr = 0.001
    classification_network = MLP(lr, True)
    acpl_network = MLP(lr, False)

    train = False # True/False whether to retrain model
    if train:
        # classification network
        class_train_set = pd.read_sql(class_train_qry, conn)
        class_test_set = pd.read_sql(class_test_qry, conn)
        
        train_X = np.array([convert_fen(x) for x in class_train_set['Input_String']])
        train_y = np.array(class_train_set['Best_Move'])

        test_X = np.array([convert_fen(x) for x in class_test_set['Input_String']])
        test_y = np.array(class_test_set['Best_Move'])

        classification_network.train(train_X, train_y, test_X, test_y)

        # acpl network
        acpl_train_set = pd.read_sql(acpl_train_qry, conn)
        acpl_test_set = pd.read_sql(acpl_test_qry, conn)

        train_X = np.array([convert_fen(x) for x in acpl_train_set['Input_String']])
        train_y = np.array(acpl_train_set['CP_Loss'].tolist()).astype(np.float)

        test_X = np.array([convert_fen(x) for x in acpl_test_set['Input_String']])
        test_y = np.array(acpl_test_set['CP_Loss'].tolist()).astype(np.float)

        acpl_network.train(train_X, train_y, test_X, test_y)

        conn.close()

        class_weights_name = os.path.join(weight_path, 'class_weights.h5')
        acpl_weights_name = os.path.join(weight_path, 'acpl_weights.h5')
        os.remove(class_weights_name)
        os.remove(acpl_weights_name)

        classification_network.model.save_weights(class_weights_name)
        acpl_network.model.save_weights(acpl_weights_name)
    else:
        conn.close()
        cat_weights_name = os.path.join(weight_path, 'class_weights.h5')
        reg_weights_name = os.path.join(weight_path, 'acpl_weights.h5')
        classification_network.model.load_weights(cat_weights_name)
        acpl_network.model.load_weights(reg_weights_name)

    #fen_test = '2rq1rk1/5pBp/p1bppnp1/1p6/4P1P1/1BN2P2/PPPQ4/2KR3R b - - 0 17' #random
    #fen_test = '5rk1/5ppp/2pbr3/1p1n3q/3P2b1/1BPQB1P1/1P1N1P1P/R3R1K1 w - - 0 20' #Marshall after 19...axb5
    fen_test = '2kr1b1r/pb1n1p2/5P2/1q1p4/Npp5/4B1P1/1P3PBP/R2Q1RK1 b - - 0 19' #Botvinnik after 19 Be3
    #fen_test = 'r1bqk2r/pppp1Npp/2n2n2/4p3/2B1P3/8/PPPP1bPP/RNBQ1K1R b kq - 0 6' #Traxler after 6 Kf1
    #fen_test = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' #starting position
    example_input = convert_fen(fen_test)
    example_input = example_input.reshape(1, 832)

    err_chance = classification_network.model.predict(example_input)
    exp_acpl = acpl_network.model.predict(example_input)

    print("Error Chance:", err_chance[0][1], "Average Error Size:", exp_acpl[0][0])

    """
    Look into a way to model different rating groups
    """


if __name__ == '__main__':
    main()