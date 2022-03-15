import datetime
import numpy as np
import tensorflow as tf
from data_processing import get_lstm_train_test_data
from data_processing import get_lstm_train_data_by_date
from data_processing import export_statistical_model_accuracy
from data_processing import write_history_stocks_probability_to_db
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


less_than = 0.2
greater_than = 0.8

lr = 3e-4
epochs = 400
batch_size = 512

# datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def get_sigmoid_lstm_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Convolution1D(64, 3, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.LSTM(32, dropout=0.2, return_sequences=True),
        tf.keras.layers.LSTM(16, dropout=0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


def get_linear_lstm_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Convolution1D(64, 3, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.LSTM(32, dropout=0.2, return_sequences=True),
        tf.keras.layers.LSTM(16, dropout=0.2),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    return model


class ConditionalMetrics(tf.keras.metrics.Metric):
    def __init__(self, name='acc', **kwargs):
        super(ConditionalMetrics, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.total_num_sample = self.add_weight(name='total', initializer='zeros')

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        less, greater = tf.less(y_pred, less_than), tf.greater(y_pred, greater_than)
        less_greater = tf.reduce_any(tf.concat([less, greater], axis=1), axis=1)

        y_true = tf.cast(y_true[less_greater], dtype=tf.bool)
        y_pred = tf.where(y_pred[less_greater] > 0.5, x=True, y=False)

        values = tf.cast(y_true == y_pred, dtype=self.dtype)

        self.true_positives.assign_add(tf.reduce_sum(values))
        self.total_num_sample.assign_add(tf.cast(tf.shape(values)[0], dtype=self.dtype))

    def result(self):
        return self.true_positives / self.total_num_sample

    def reset_states(self):
        self.true_positives.assign(0.0)
        self.total_num_sample.assign(0.0)


def train_dobule_model():
    less_than = 0.5
    greater_than = 0.5
    
    log_path = 'logs/000300_v10'
    checkpoint_path = "weight/000300_v10"

    train_summary_writer = tf.summary.create_file_writer(log_path + '/train')
    test_summary_writer = tf.summary.create_file_writer(log_path + '/test')
    
    # target_train_summary_writer = tf.summary.create_file_writer(log_path + '/target_train')
    # target_test_summary_writer = tf.summary.create_file_writer(log_path + '/target_test')

    # x_train, x_test, y_train, y_test = get_lstm_train_test_data('000300/*.csv', data_slice=slice(0, 1))
    x_train, x_test, y_train, y_test = get_pre_train_data(r'datasets\000300_v10.csv', test_date='2018-12-31')

    # train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
    # test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    model = get_sigmoid_lstm_model()
    target_model = get_sigmoid_lstm_model()

    model.build(input_shape=(None, x_train.shape[1], x_train.shape[2]))
    target_model.build(input_shape=(None, x_train.shape[1], x_train.shape[2]))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy', metrics=[ConditionalMetrics()])
    target_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy', metrics=[ConditionalMetrics()])
    
    for epoch in range(epochs):
        
        # for i in range(0, len(x_train), batch_size):
        #     b_x_train = x_train[i:i+batch_size]
        #     b_y_train = y_train[i:i+batch_size]
        #     target_model.train_on_batch(b_x_train, b_y_train)
    
        # train_loss, train_acc = target_model.evaluate(x_train, y_train, verbose=0)
        # test_loss, test_acc = target_model.evaluate(x_test, y_test, verbose=0)
        # print(f'epoch: {epoch+1} \t loss: {train_loss} \t acc: {train_acc} \t val_loss: {test_loss} \t val_acc: {test_acc}')
                
        # with target_train_summary_writer.as_default():
        #     tf.summary.scalar('loss', train_loss, step=epoch)
        #     tf.summary.scalar('accuracy', train_acc, step=epoch)
        # with target_test_summary_writer.as_default():
        #     tf.summary.scalar('loss', test_loss, step=epoch)
        #     tf.summary.scalar('accuracy', test_acc, step=epoch)
        
        # if epoch != 0 and epoch % 10 == 0:
        #     target_model.save_weights(checkpoint_path + '/target_lstm/lstm.ckpt')
        
        pred_probs = target_model.predict(x_train)
        less_greater = np.squeeze((pred_probs > greater_than) | (pred_probs < less_than))
        
        if not (greater_than == 0.8 and less_than == 0.2):
            greater_than += 0.001
            less_than -= 0.001
        
        
        if any(less_greater):
        
            _x_train = x_train[less_greater]
            _y_train = y_train[less_greater]
            
            for j in range(0, len(_x_train), batch_size):
                _b_x_train = _x_train[j:j+batch_size]
                _b_y_train = _y_train[j:j+batch_size]
                model.train_on_batch(_b_x_train, _b_y_train)
               
            train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
            print(f'epoch: {epoch+1} \t _loss: {train_loss} \t _acc: {train_acc} \t _val_loss: {test_loss} \t _val_acc: {test_acc}\n')

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss, step=epoch)
                tf.summary.scalar('accuracy', train_acc, step=epoch)
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss, step=epoch)
                tf.summary.scalar('accuracy', test_acc, step=epoch)

            if epoch != 0 and epoch % 10 == 0:
                model.save_weights(checkpoint_path + '/lstm/lstm.ckpt')

        if epoch != 0 and epoch % 20 == 0:
            target_model.set_weights(model.get_weights())


def train_standard():
    log_path = 'logs/000300_v12'
    checkpoint_path = 'weight/000300_v12/lstm.ckpt'

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True)

    # x_train, x_test, y_train, y_test = get_lstm_train_test_data('000300/*.csv', data_slice=slice(0, 1))
    x_train, x_test, y_train, y_test = get_pre_train_data(r'datasets\000300_v12.csv', test_date='2018-12-31', threshold_slope=0.005)

    # train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
    # test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)


    model = get_sigmoid_lstm_model()
    model.build(input_shape=(None, x_train.shape[1], x_train.shape[2]))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy', metrics=[ConditionalMetrics()])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs, callbacks=[tb_callback, cp_callback])


def train_split_date(name, dir, future_steps, source_path='000300/*.csv', data_slice=None):
    # x_train, _, y_train, _ = get_pre_train_data(rf'datasets\{dir}.csv', source_path=source_path, test_date='2018-12-31', future_steps=future_steps, data_slice=data_slice, classify_num=1)

    # model = get_sigmoid_lstm_model()
    model = get_linear_lstm_model()
    # model.build(input_shape=(None, x_train.shape[1], x_train.shape[2]))
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss='huber', metrics=['mae'])
    
    # model.fit(x_train, y_train, batch_size=512, epochs=400)
    # model.save_weights(rf'weight/{dir}/before_2018/lstm.ckpt')
    
    model.load_weights(rf'weight/{dir}/before_2018/lstm.ckpt')
    
    write_history_stocks_probability_to_db('2019-01-01', '2019-04-01', stock_path=source_path, model_dir={name :rf'weight/{dir}/before_2018'})

    dates = [
        ('2019-01-01', '2019-04-01'),
        ('2019-04-01', '2019-07-01'),
        ('2019-07-01', '2019-10-01'),
        ('2019-10-01', '2020-01-01'),
        ('2020-01-01', '2020-04-01'),
        ('2020-04-01', '2020-07-01'),
        ('2020-07-01', '2020-10-01'),
        ('2020-10-01', '2021-01-01'),
        ('2021-01-01', '2021-04-01'),
        ('2021-04-01', '2021-07-01'),
        ('2021-07-01', '2021-10-01'),
        ('2021-10-01', '2022-01-01')
    ]

    for i in range(len(dates)):
        start, end = dates[i]

        window_x_test, window_y_test = get_pre_train_data_by_date(rf'datasets\{dir}.csv', start, end, future_steps=future_steps, classify_num=1)
        model.fit(window_x_test, window_y_test, batch_size=128, epochs=50)
        model.save_weights(rf'weight/{dir}/{start}_{end}/lstm.ckpt')
        
        if i == len(dates) - 1: break
        
        next_start, next_end = dates[i+1]
        write_history_stocks_probability_to_db(next_start, next_end,  model_dir={name: rf'weight/{dir}/{start}_{end}'})


def train_spole():
    x_train, x_test, y_train, y_test = get_pre_train_data(r'datasets\000300_v19.csv', test_date='2018-12-31', future_steps=10, classify_num=1)

    model = get_linear_lstm_model()
    model.build(input_shape=(None, x_train.shape[1], x_train.shape[2]))
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-5), loss='huber', metrics=['mae'])
    
    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=512, epochs=3000)
    model.save_weights(rf'weight/000300_v19/lstm.ckpt')


def train():
    
    x_train, x_test, y_train, y_test = get_pre_train_data(r'datasets\000300_v23.csv', test_date='2018-12-31', future_steps=3, time_steps=60)

    model = get_sigmoid_lstm_model()
    model.build(input_shape=(None, x_train.shape[1], x_train.shape[2]))
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss='binary_crossentropy', metrics=[ConditionalMetrics()])
    
    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=128, epochs=1000)
    model.save_weights(r'weight/000300_v23/lstm.ckpt')


def train_all_factors():
    
    log_path = 'logs/000300_v24'
    checkpoint_path = "weight/000300_v24/lstm.ckpt"

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True)
    
    x_train, x_test, y_train, y_test = get_pre_train_data(r'datasets\000300_v24.csv', time_steps=60, test_date='2018-12-31', source_path='000300_all_factors/*.csv')

    model = tf.keras.Sequential([
        tf.keras.layers.Convolution1D(128, 3, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Convolution1D(64, 3, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64, dropout=0.2, return_sequences=True),
        tf.keras.layers.LSTM(32, dropout=0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    def scheduler(epoch, lr):
        if epoch < 500:
            return 3e-4
        elif epoch < 1000:
            return 1e-4
        else:
            return 3e-5

    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    
    model.build(input_shape=(None, x_train.shape[1], x_train.shape[2]))
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss='binary_crossentropy', metrics=[ConditionalMetrics()])
    
    # model.save_weights(rf'weight/000300_v23/lstm.ckpt')
    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=128, epochs=2000, callbacks=[tb_callback, cp_callback, lr_callback])


def quick_train(model_name, time_steps, future_steps, start_date='2020-01-01', end_date='2020-12-31', csv=r'models\quick_test.csv', source_path=r'000300_all_factors/*.csv'):
    
    x_train, y_train = get_lstm_train_data_by_date(csv, start_date=start_date, end_date=end_date, time_steps=time_steps, future_steps=future_steps, source_path=source_path)

    model = tf.keras.Sequential([
        tf.keras.layers.Convolution1D(128, 3, activation='relu', kernel_regularizer='l2'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64, dropout=0.2, return_sequences=True, kernel_regularizer='l2'),
        tf.keras.layers.LSTM(32, dropout=0.2, kernel_regularizer='l2'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch != 0 and epoch % 200 == 0:
                model.save(rf"models/{model_name}_{epoch}")

    
    model.build(input_shape=(None, x_train.shape[1], x_train.shape[2]))
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss='binary_crossentropy', metrics=['acc'])
    
    model.fit(x_train, y_train, batch_size=128, epochs=600, callbacks=[CustomCallback()])


def quick_test(model_name, time_steps, future_steps, csv, source_path, m):
    
    for epoch in [200, 400]:
        model = tf.keras.models.load_model(rf"models/{model_name}_{epoch}")
        model_dir = {f'{m}_{epoch}': ''}
        write_history_stocks_probability_to_db('2021-01-01', '2021-12-31', time_steps=time_steps, model_dir=model_dir, csv_name=csv, stock_path=source_path, model=model)
        export_statistical_model_accuracy('2021-01-01', '2021-12-31', m=[f'{m}_{epoch}'], avg_m=[], level='year', future_steps=future_steps, stock_path=source_path, writer_dir=rf'models/{m}_{epoch}')


if __name__ == '__main__':
    
    # quick_train(model_name='quick_test_all_t30_f10', time_steps=30, future_steps=10, csv=r'models\quick_test_all.csv', source_path=r'000300_all_factors/*.csv')
    # quick_test('quick_test_all_t30_f10', 30, 10, r'models\quick_test_all.csv', r'000300_all_factors/*.csv', 'all_t30_f10')
    # quick_train(model_name='quick_test_all_t30_f5', time_steps=30, future_steps=5, csv=r'models\quick_test_all.csv', source_path=r'000300_all_factors/*.csv')
    # quick_test('quick_test_all_t30_f5', 30, 5, r'models\quick_test_all.csv', r'000300_all_factors/*.csv', 'all_t30_f5')
    # quick_train(model_name='quick_test_all_t60_f10', time_steps=60, future_steps=10, csv=r'models\quick_test_all.csv', source_path=r'000300_all_factors/*.csv')
    # quick_test('quick_test_all_t60_f10', 60, 10, r'models\quick_test_all.csv', r'000300_all_factors/*.csv', 'all_t60_f10')
    # quick_train(model_name='quick_test_all_t60_f5', time_steps=60, future_steps=5, csv=r'models\quick_test_all.csv', source_path=r'000300_all_factors/*.csv')
    # quick_test('quick_test_all_t60_f5', 60, 5, r'models\quick_test_all.csv', r'000300_all_factors/*.csv', 'all_t60_f5')
    
    # quick_train(model_name='quick_test_t30_f10', time_steps=30, future_steps=10, csv=r'models\quick_test.csv', source_path=r'000300/*.csv')
    # quick_train(model_name='quick_test_t30_f5', time_steps=30, future_steps=5, csv=r'models\quick_test.csv', source_path=r'000300/*.csv')
    # quick_train(model_name='quick_test_t60_f10', time_steps=60, future_steps=10, csv=r'models\quick_test.csv', source_path=r'000300/*.csv')
    # quick_train(model_name='quick_test_t60_f5', time_steps=60, future_steps=5, csv=r'models\quick_test.csv', source_path=r'000300/*.csv')
    
    
    # train()
    
    # train_all_factors()
    
    # ()
    
    # train_split_date('v15', '000300_v15', 10)
    
    # train_split_date('v16', '000300_v16', 5)
    
    # train_split_date('v18', '000300_v18', 10, data_slice=slice(0, 100), source_path='datasets/factors_datas/*.csv')
    
    # train_split_date('v17', '000300_v17', 10, data_slice=slice(0, 100))
    
    # train_split_date('v19', '000300_v19', 10, data_slice=slice(0, 100))
        
        
    # x_train, x_test, y_train, y_test = get_lstm_train_test_data('000300/*.csv', data_slice=slice(0, 1))

    # train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
    # test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    # def scheduler(epoch, lr):
    #     if epoch < 300:
    #         return 3e-4
    #     elif epoch < 200:
    #         return 1e-4
    #     else:
    #         return 3e-5


    # lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    # class Accuracy(tf.keras.callbacks.Callback):
    #     def on_epoch_end(self, epoch, logs=None):
    #         f_y_true, f_y_pred, y_pred = [], [], []
    #         y_true = tf.cast(y_test, tf.float32)
    #         for i in range(0, len(x_test), 128):
    #             y_pred.extend(model(x_test[i:i+128]))
    #         y_pred = tf.squeeze(y_pred)
    #         for i, v in enumerate(y_pred):
    #             if v > less_than and v < greater_than:
    #                 continue
    #             f_y_true.append(y_true[i])
    #             f_y_pred.append(0 if v < 0.5 else 1)
    #         f_y_true = tf.cast(f_y_true, tf.bool)
    #         f_y_pred = tf.cast(f_y_pred, tf.bool)
    #         values = tf.cast(tf.equal(f_y_true, f_y_pred), tf.float32)
    #         acc = 0 if len(f_y_pred) == 0 else (tf.reduce_sum(values) / len(f_y_pred)).numpy()
    #         print('\t less {} and greater {} acc: {}'.format(less_than, greater_than, acc))
    print("====complete====")