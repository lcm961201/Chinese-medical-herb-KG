# coding=utf-8
import keras
import numpy as np
from bilstm_crf_model import BiLstmCrfModel
from crf_layer import CRF
from data_helpers import NerDataProcessor

max_len = 80
vocab_size = 2410
embedding_dim = 200
lstm_units = 128

if __name__ == '__main__':
    ndp = NerDataProcessor(max_len,vocab_size)
    '''训练集'''
    train_X,train_y = ndp.read_data(
            'D:\研究生\第一篇-知识图谱\实验\BiLSTM-CRF\伤寒论\char_token12568.txt',
            is_training_data=True
        )
    train_X,train_y = ndp.encode(train_X,train_y)
    # '''验证集'''
    # dev_X,dev_y = ndp.read_data(
    #         "../../../ChineseBLUE/data/cMedQANER/dev.txt",
    #         is_training_data=False
    #     )
    # dev_X,dev_y = ndp.encode(dev_X,dev_y)
    '''测试集'''
    test_X,test_y = ndp.read_data(
            'D:\研究生\第一篇-知识图谱\实验\BiLSTM-CRF\伤寒论\char_token910.txt',
            is_training_data=False
        )
    test_X,test_y = ndp.encode(test_X,test_y)

    class_nums = ndp.class_nums
    word2id = ndp.word2id
    tag2id = ndp.tag2id
    id2tag = ndp.id2tag
    import pickle
    '''保存上面三行中的序列'''
    pickle.dump(
            (word2id,tag2id,id2tag),
            open('./checkpoint/word_tag_id1.pkl','wb')
        )

    bilstm_crf = BiLstmCrfModel(
            max_len,
            vocab_size,
            embedding_dim,
            lstm_units,
            class_nums
        )
    model = bilstm_crf.build()

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=4, 
        verbose=1)

    earlystop = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        verbose=2, 
        mode='min'
        )
    bast_model_filepath = 'D:\研究生\第一篇-知识图谱\实验\BiLSTM-CRF\checkpoint\\best_bilstm_crf_model1.h5'
    checkpoint = keras.callbacks.ModelCheckpoint(
        bast_model_filepath, 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True,
        mode='min'
        )
    model.fit(
        x=train_X,
        y=train_y,
        batch_size=32,
        epochs=80,
        # validation_data=(dev_X, dev_y),
        shuffle=True,
        callbacks=[reduce_lr,earlystop,checkpoint]
        )
    # model.load_weights(bast_model_filepath)
    model.save('./checkpoint/bilstm_crf_model.h5')

    '''预测'''
    pred = model.predict(test_X)
    # print(test_y)
    # print(pred)

# '''模型评估部分'''
    from metrics import *
    y_true, y_pred = [],[]

    for t_oh,p_oh in zip(test_y,pred):
        t_oh = np.argmax(t_oh,axis=1)
        t_oh = [id2tag[i].replace('_','-') for i in t_oh if i!=0]
        p_oh = np.argmax(p_oh,axis=1)
        p_oh = [id2tag[i].replace('_','-') for i in p_oh if i!=0]
        y_true.append(t_oh)
        y_pred.append(p_oh)
        # print(y_pred)
        # '''将预测结果写入文档'''
        # with open('./pred_result.txt','w',encoding='utf-8') as f:
        #     for i in y_pred:
        #         f.write(i)

    f1 = f1_score(y_true,y_pred,suffix=False)
    p = precision_score(y_true,y_pred,suffix=False)
    r = recall_score(y_true,y_pred,suffix=False)
    acc = accuracy_score(y_true,y_pred)
    print("f1_score: {:.4f}, precision_score: {:.4f}, recall_score: {:.4f}, accuracy_score: {:.4f}".format(f1,p,r,acc))
    print(classification_report(y_true, y_pred, digits=4, suffix=False))
