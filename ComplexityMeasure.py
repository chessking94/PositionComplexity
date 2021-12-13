import os
import pyodbc as sql
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras import Sequential 
import matplotlib.pyplot as plt

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
    #position_matrix[12][5] = inp_str.split('|')[2] # rating
    #position_matrix[12][6] = inp_str.split('|')[1] # number of candidate moves; currently set to only count moves within 100 CP of best move
    #for i in range(int(inp_str.split('|')[1])): # set number of elements equal to number of candidate moves; up to 32 candidate moves (through [12][36])
    #    position_matrix[12][6+i] = 1
    # position_matrix[12][7] through position_matrix[12][63] still unused

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

lr = 0.001

class MLP:
    def __init__(self, learning_rate, classification): # Input - learning_rate, boolean classification [T] or regression [F]
        self.learning_rate = learning_rate
        if classification:
            self.model = self.build_classification_model()
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                            loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
        else:
            self.model = self.build_regression_model()
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                                    loss='mse', metrics=['mae', 'mse'])

    def build_classification_model(self):
        model = Sequential()
        
        model.add(Dense(64, input_dim=832))
        model.add(Activation('relu'))

        model.add(Dense(15))
        model.add(Activation('relu'))

        model.add(Dense(2))
        model.add(Activation('softmax'))

        """        
        model.add(Dense(1048, input_dim=832))
        model.add(Activation('relu'))
        #model.add(Dropout(0.2))

        model.add(Dense(500))
        model.add(Activation('relu'))
        #model.add(Dropout(0.2))

        model.add(Dense(50))
        model.add(Activation('relu'))
        #model.add(Dropout(0.2))

        model.add(Dense(2))
        model.add(Activation('softmax'))
        """

        #model.summary()
        return model

    def build_regression_model(self):
        model = Sequential()
        
        
        model.add(Dense(64, input_dim=832))
        model.add(Activation('relu'))

        #model.add(Dense(15))
        #model.add(Activation('relu'))

        model.add(Dense(1))
        model.add(Activation('relu'))
        
        """
        model.add(Dense(1048, input_dim=832))
        model.add(Activation('relu'))
        #model.add(Dropout(0.2))

        model.add(Dense(500))
        model.add(Activation('relu'))
        #model.add(Dropout(0.2))

        model.add(Dense(50))
        model.add(Activation('relu'))
        #model.add(Dropout(0.2))

        model.add(Dense(1))
        """        

        #model.summary()
        return model

    def train(self, x, y, val_x, val_y):
        history = self.model.fit(x, y, epochs=1, verbose=1, validation_data=(val_x, val_y))
        return history

def main():
    conn = sql.connect('Driver={ODBC Driver 17 for SQL Server};Server=HUNT-PC1;Database=ChessAnalysis;Trusted_Connection=yes;')    
    train_qry = '''
SELECT
CONCAT(m.FEN, '|', m.CandidateMoves, '|', CASE m.Color WHEN 'White' THEN g.WhiteElo ELSE g.BlackElo END) AS Input_String,
m.CP_Loss,
CASE m.Move_Rank WHEN 1 THEN 1 ELSE 0 END AS Best_Move
FROM ControlMoves m
JOIN ControlGames g ON m.GameID = g.GameID
WHERE g.CorrFlag = 0
AND m.GameID <= 20000
AND m.CP_Loss IS NOT NULL
AND CAST(m.T1_Eval AS decimal) <= 3.5
AND m.IsTheory = 0
'''
    test_qry = '''
SELECT
CONCAT(m.FEN, '|', m.CandidateMoves, '|', CASE m.Color WHEN 'White' THEN g.WhiteElo ELSE g.BlackElo END) AS Input_String,
m.CP_Loss,
CASE m.Move_Rank WHEN 1 THEN 1 ELSE 0 END AS Best_Move
FROM ControlMoves m
JOIN ControlGames g ON m.GameID = g.GameID
WHERE g.CorrFlag = 0
AND m.GameID > 20000
AND m.CP_Loss IS NOT NULL
AND CAST(m.T1_Eval AS decimal) <= 3.5
AND m.IsTheory = 0
'''
    
    weight_path = r'C:\Users\eehunt\Repository\PositionComplexity\weights'

    classification_network = MLP(lr, True)
    acpl_network = MLP(lr, False)

    train = False # True/False whether to retrain model
    if train:
        train_set = pd.read_sql(train_qry, conn)
        test_set = pd.read_sql(test_qry, conn)
        conn.close()
        
        train_X = np.array([convert_fen(x) for x in train_set['Input_String']])
        train_y_best = np.array(train_set['Best_Move'])
        train_y_cp = np.array(train_set['CP_Loss'].tolist()).astype(np.float)

        test_X = np.array([convert_fen(x) for x in test_set['Input_String']])
        test_y_best = np.array(test_set['Best_Move'].tolist())
        test_y_cp = np.array(test_set['CP_Loss'].tolist()).astype(np.float)

        classification_network.train(train_X, train_y_best, test_X, test_y_best)
        acpl_network.train(train_X, train_y_cp, test_X, test_y_cp)

        cat_weights_name = os.path.join(weight_path, 'cat_model_weights.h5')
        reg_weights_name = os.path.join(weight_path, 'reg_model_weights.h5')
        os.remove(cat_weights_name)
        os.remove(reg_weights_name)

        classification_network.model.save_weights(cat_weights_name)
        acpl_network.model.save_weights(reg_weights_name)
    else:
        conn.close()
        cat_weights_name = os.path.join(weight_path, 'cat_model_weights.h5')
        reg_weights_name = os.path.join(weight_path, 'reg_model_weights.h5')
        classification_network.model.load_weights(cat_weights_name)
        acpl_network.model.load_weights(reg_weights_name)

    #fen_test = '2rq1rk1/5pBp/p1bppnp1/1p6/4P1P1/1BN2P2/PPPQ4/2KR3R b - - 0 17' #random
    fen_test = '5rk1/5ppp/2pbr3/1p1n3q/3P2b1/1BPQB1P1/1P1N1P1P/R3R1K1 w - - 0 20' #Marshall after 19...axb5
    #fen_test = '2kr1b1r/pb1n1p2/5P2/1q1p4/Npp5/4B1P1/1P3PBP/R2Q1RK1 b - - 0 19' #Botvinnik after 19 Be3
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