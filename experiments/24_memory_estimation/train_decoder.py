import os
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras.callbacks import CSVLogger,EarlyStopping
from tensorflow.keras import layers

# Global variables
parent_dir = os.path.realpath(os.path.dirname(__file__))

def make_model(inp_dim,out_dim):
    model = keras.Sequential()
    model.add(layers.Dense(500, activation='relu', input_dim=inp_dim))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(out_dim, activation='linear')) # softmax
    
    # Compile the model
    model.compile(optimizer='adam', 
                  loss='mse', # 'categorical_crossentropy'
                  metrics=['mse'])

    return model

def train_decoder(append_logs=False):
    results = {}
    for id in ['base2', 'base4']:

        M = np.load(os.path.join(parent_dir, f"M-{id}.npy"))
        X = np.load(os.path.join(parent_dir, f"X-{id}.npy"))
        Y = np.load(os.path.join(parent_dir, f"Y-{id}.npy"))
        
        #Y_unique = np.unique(Y)
        #Y_cat = (Y_unique == Y).astype(float)
        #Y = Y_cat

        scaler_Y = StandardScaler()
        scaler_Y.fit(Y)
        Y = scaler_Y.transform(Y)

        scaler_M = StandardScaler()
        scaler_M.fit(M)

        M_train, M_test, Y_train, Y_test = train_test_split(
            scaler_M.transform(M), Y, test_size=0.1, random_state=1)

        model = make_model(inp_dim=M.shape[-1],out_dim=Y.shape[-1])

        csv_logger = CSVLogger(os.path.join(parent_dir,f"log-{id}.csv"), 
                               append=append_logs, separator=' ')
        es_cb = EarlyStopping(monitor="val_loss", patience=50)
        model.fit(M_train, Y_train, 
                  validation_data=(M_test, Y_test),
                  epochs=500, batch_size=M_train.shape[0] // 128,
                  callbacks=[csv_logger])
        model.save(os.path.join(parent_dir, f"model-{id}"))

        y_pred = model.predict(M_test)
        err = (y_pred - Y_test)
        err_file = os.path.join(parent_dir,f"err-{id}.csv")
        with open(err_file, 'a' if append_logs else 'w') as fp:
            fp.write(" ".join([str(v) for v in np.mean(err**2,axis=0)]) + "\n")

        results[id] = np.mean(err**2)
    
    return results

if __name__ == '__main__':
    results = train_decoder()   
    
    for k in results.keys():
        print(f"{k}: {results[k]}")

    print("FIN")

        
        