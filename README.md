# model v1
model = tf.keras.Sequential([
    tf.keras.layers.Convolution1D(64, 3, activation='relu', input_shape=(x_train.shape[-2], x_train.shape[-1])),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(32, dropout=0.2, return_sequences=True),
    tf.keras.layers.LSTM(32, dropout=0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
1. lr: 3e-4
2. epochs: 413
3. batch_size: 128


# model v2
model = tf.keras.Sequential([
    tf.keras.layers.Convolution1D(64, 3, activation='relu', input_shape=(x_train.shape[-2], x_train.shape[-1])),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(32, dropout=0.2, return_sequences=True),
    tf.keras.layers.LSTM(32, dropout=0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
1. lr: if epochs < 500 3e-4 else 3e-5
2. epochs: 1000
3. batch_size: 128


# model v3
model = tf.keras.Sequential([
    tf.keras.layers.Convolution1D(64, 3, padding='same', activation='relu', input_shape=(x_train.shape[-2], x_train.shape[-1])),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Convolution1D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32, dropout=0.2, return_sequences=True),
    tf.keras.layers.LSTM(32, dropout=0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
1. lr: if epochs < 500 3e-4 else 3e-5
2. epochs: 800
3. batch_size: 128


# model v4
model = tf.keras.Sequential([
    tf.keras.layers.Convolution1D(64, 3, padding='same', activation='relu', input_shape=(x_train.shape[-2], x_train.shape[-1])),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(32, dropout=0.2, return_sequences=True),
    tf.keras.layers.LSTM(32, dropout=0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
1. lr: if epochs < 300 3e-4 elif epochs < 200 1e-4 else 3e-5
2. epochs: 600
3. batch_size: 128


# model v5 —— 99 stocks 311 factors 
model = tf.keras.Sequential([
    tf.keras.layers.Convolution1D(128, 3, padding='same', activation='relu', input_shape=(x_train.shape[-2], x_train.shape[-1])),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(64, dropout=0.2, return_sequences=True),
    tf.keras.layers.LSTM(32, dropout=0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
1. lr: 3e-4
2. epochs: 400
3. batch_size: 128


# model v6 —— 99 stocks 81 factors 
model = tf.keras.Sequential([
    tf.keras.layers.Convolution1D(128, 3, padding='same', activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal(), input_shape=(x_train.shape[-2], x_train.shape[-1])),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(64, dropout=0.2, return_sequences=True, kernel_initializer=tf.keras.initializers.GlorotNormal()),
    tf.keras.layers.LSTM(32, dropout=0.2, kernel_initializer=tf.keras.initializers.GlorotNormal()),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
1. lr: 3e-4
2. epochs: 400
3. batch_size: 128


# model v7 —— 99 stocks 81 factors
model = tf.keras.Sequential([
    tf.keras.layers.Convolution1D(128, 3, padding='same', activation='relu', input_shape=(x_train.shape[-2], x_train.shape[-1])),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(64, dropout=0.2, return_sequences=True),
    tf.keras.layers.LSTM(32, dropout=0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
1. lr: 3e-4
2. epochs: 400
3. batch_size: 128


# model v8 —— 99 stocks backtrader
model = tf.keras.Sequential([
    tf.keras.layers.Convolution1D(128, 3, padding='same', activation='relu', input_shape=(x_train.shape[-2], x_train.shape[-1])),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(64, dropout=0.2, return_sequences=True),
    tf.keras.layers.LSTM(32, dropout=0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
1. lr: 3e-4
2. epochs: 400
3. batch_size: 128

# model v9
model = tf.keras.Sequential([
    tf.keras.layers.Convolution1D(64, 3, padding='same', activation='relu', input_shape=(x_train.shape[-2], x_train.shape[-1])),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(32, dropout=0.2, return_sequences=True),
    tf.keras.layers.LSTM(32, dropout=0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
1. lr: 3e-4
2. epochs: 2000
3. batch_size: 128

# model v10 double

# model v11 threshold_slope=0.008
model = tf.keras.Sequential([
    tf.keras.layers.Convolution1D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(64, dropout=0.2, return_sequences=True),
    tf.keras.layers.LSTM(32, dropout=0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
1. lr: 3e-4
2. epochs: 400
3. batch_size: 128


# model v12 threshold_slope=0.008
model = tf.keras.Sequential([
    tf.keras.layers.Convolution1D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(64, dropout=0.2, return_sequences=True),
    tf.keras.layers.LSTM(32, dropout=0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
1. lr: 3e-4
2. epochs: 2000
3. batch_size: 128


# model v13 threshold_slope=0.005 分季度训练
model = tf.keras.Sequential([
    tf.keras.layers.Convolution1D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(64, dropout=0.2, return_sequences=True),
    tf.keras.layers.LSTM(32, dropout=0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
1. lr: 3e-4
2. epochs: 400
3. batch_size: 128


# model v14 threshold_slope=0.005 分季度训练
model = tf.keras.Sequential([
    tf.keras.layers.Convolution1D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(64, dropout=0.2, return_sequences=True),
    tf.keras.layers.LSTM(32, dropout=0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
1. lr: 3e-4
2. epochs: 5000 --- 400
3. batch_size: 128


# model v15 分季度训练 future_steps=10
model = tf.keras.Sequential([
    tf.keras.layers.Convolution1D(64, 3, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(32, dropout=0.2, return_sequences=True),
    tf.keras.layers.LSTM(16, dropout=0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
return model
1. lr: 3e-4
2. epochs: 1000 --- 50
3. batch_size: 512 --- 128


# model v16 分季度训练 future_steps=5
model = tf.keras.Sequential([
    tf.keras.layers.Convolution1D(64, 3, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(32, dropout=0.2, return_sequences=True),
    tf.keras.layers.LSTM(16, dropout=0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
return model
1. lr: 3e-4
2. epochs: 1000 --- 50
3. batch_size: 512 --- 128


v20 ==> 预测一天

v21 ==> 预测三天

v23 ==> 预测三天 使用60天历史