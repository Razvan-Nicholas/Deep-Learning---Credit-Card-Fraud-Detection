from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_mlp(input_dim, layers=[32, 16], lr=0.001):
    model = Sequential()
    for i, units in enumerate(layers):
        model.add(Dense(units, activation='relu', input_dim=input_dim if i==0 else None))
        model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['Recall'])
    return model
