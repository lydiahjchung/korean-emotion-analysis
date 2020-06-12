from preprocess import Preprocess
from data_split import Splitting
from layers import *
from tensorflow import keras
from tensorflow.keras import layers

if __name__ == "__main__":
    preprocess = Preprocess()
    splitting = Splitting()

    ''' preprocessing '''
    # open and save input file - labeled text data
    with open('../../Data/labeled_final.txt') as f:
        input_data = f.readlines()

    # splitting input_data into sentences and emotion labels
    labeled = []
    for line in input_data:
        cut_idx = len(line) - line[::-1].find(';')
        sentence, emotion = line[:cut_idx-1], line[cut_idx:].strip()
        labeled.append([sentence, emotion])

    # making list of preprocessed labeled data
    processed_input = preprocess.preprocessing_train(labeled)

    # integer encoding the preprocessed sentences in labeled data
    encoded = []
    for sentence, emotion in processed_input:
        encoded.append([preprocess.text_to_encoding(sentence), emotion])


    ''' data split '''
    # splitting the data by each label
    splitting.each_label(encoded)

    # shuffling data and splitting it into train, val
    splitting.shuffle()

    # splitting the data into x_train, y_train, x_val, y_val
    x_train, y_train, x_val, y_val = splitting.train_and_test()

    # padding sentences
    x_train = splitting.padding(x_train)
    x_val = splitting.padding(x_val)


    ''' transformer model '''
    # embedding size for each token
    embed_dim = 32
    # number of attention heads
    num_heads = 2
    # hidden layer size in feed forward network
    ff_dim = 32
    # maximum length of each sentence
    max_len = splitting.max_len

    # implementing layers
    inputs = keras.Input(shape=(max_len, ), )
    embedding_layer = TokenAndPositionEmbedding(max_len, len(preprocess.vocab)+1, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(7, activation='softmax')(x)

    # transformer model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # summary of the model
    model.summary()

    # training and fitting the model to the data
    early_stopping = keras.callbacks.EarlyStopping()
    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val), callbacks=[early_stopping]
    )


    ''' saving output of test data '''
    # opening three test data sets
    with open('../../Data/google_final_test.txt') as g:
        google_input = g.readlines()
    with open('../../Data/kakao_final_test.txt') as k:
        kakao_input = k.readlines()
    with open('../../Data/papago_final_test.txt') as p:
        papago_input = p.readlines()

    # preprocess input test data sets using vocab built by the training data set
    google_processed = preprocess.preprocessing_test(google_input)
    kakao_processed = preprocess.preprocessing_test(kakao_input)
    papago_processed = preprocess.preprocessing_test(papago_input)

    # pad each processed data sets
    google_processed = splitting.padding(google_processed)
    kakao_processed = splitting.padding(kakao_processed)
    papago_processed = splitting.padding(papago_processed)

    # predict labels and save the results into csv files
    google_predict = model.predict(google_processed)
    kakao_predict = model.predict(kakao_processed)
    papago_predict = model.predict(papago_processed)

    # determine predicted label and the probability of the prediction
    google_sentence, google_label, google_probability = splitting.predictions(google_input, google_predict)
    kakao_sentence, kakao_label, kakao_probability = splitting.predictions(kakao_input, kakao_predict)
    papago_sentence, papago_label, papago_probability = splitting.predictions(papago_input, papago_predict)

    # saving each outputs as csv
    preprocess.output_csv(google_sentence, google_label, google_probability, 'google')
    preprocess.output_csv(kakao_sentence, kakao_label, kakao_probability, 'kakao')
    preprocess.output_csv(papago_sentence, papago_label, papago_probability, 'papago')
