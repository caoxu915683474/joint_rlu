import tensorflow as tf
from utils.my_metrics import *
from utils.preprocess import *
from mode.model import Model
from tensorflow.python import debug as tf_debug

input_steps = 50
embedding_size = 64
hidden_size = 100
n_layers = 2
batch_size = 128
vocab_size = 871
slot_size = 122
intent_size = 22
epoch_num = 50

def get_model():
    model = Model(input_steps, embedding_size, hidden_size, vocab_size, slot_size, intent_size, epoch_num, batch_size, n_layers)
    model.build()
    return model

def train(is_debug=False):
    model = get_model()
    sess = tf.Session()
    if is_debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan())
    sess.run(tf.global_variables_initializer())
    train_data = open("../dataset/atis-2.train.w-intent.iob", "r").readlines()
    test_data = open("../dataset/atis-2.dev.w-intent.iob", "r").readlines()
    train_data_ed = data_pre(train_data)
    test_data_ed = data_pre(test_data)
    word2index, index2word, slot2index, index2slot, intent2index, index2intent = \
        get_info_from_training_data(train_data_ed)
    index_train = to_index(train_data_ed, word2index, slot2index, intent2index)
    index_test = to_index(test_data_ed, word2index, slot2index, intent2index)
    for epoch in range(epoch_num):
        mean_loss = 0.0
        train_loss = 0.0
        f_i = 0
        for i, batch in enumerate(getBatch(batch_size, index_train)):
            # 执行一个batch的训练
            _, loss, decoder_prediction, intent, mask, slot_W = model.step(sess, "train", batch)
            mean_loss += loss
            train_loss += loss
            if i % 30 == 0:
                if i > 0:
                    mean_loss = mean_loss / 30.0
                print('Average train loss at epoch %d, step %d: %f' % (epoch, i, mean_loss))
                mean_loss = 0
            f_i = i
        train_loss /= (f_i + 1)
        print("[Epoch {}] Average train loss： {}".format(epoch, train_loss))

        pred_slots = []
        slot_accs = []
        intent_accs = []
        for j, batch in enumerate(getBatch(batch_size, index_test)):
            decoder_prediction, intent = model.step(sess, "test", batch)
            decoder_prediction = np.transpose(decoder_prediction, [1, 0])
            if j == 0:
                index = random.choice(range(len(batch)))
                sen_len = batch[index][1]
                print("Input Sentence       : ", index_seq2word(batch[index][0], index2word)[:sen_len])
                print("Slot Truth           : ", index_seq2slot(batch[index][2], index2slot)[:sen_len])
                print("Slot Prediction      : ", index_seq2slot(decoder_prediction[index], index2slot)[:sen_len])
                print("Intent Truth         : ", index2intent[batch[index][3]])
                print("Intent Prediction    : ", index2intent[intent[index]])
            slot_pred_length = list(np.shape(decoder_prediction))[1]
            pred_padded = np.lib.pad(decoder_prediction, ((0, 0), (0, model.input_steps - slot_pred_length)), mode="constant", constant_values=0)
            pred_slots.append(pred_padded)
            true_slot = np.array((list(zip(*batch))[2]))
            true_length = np.array((list(zip(*batch))[1]))
            true_slot = true_slot[:, :slot_pred_length]
            slot_acc = accuracy_score(true_slot, decoder_prediction, true_length)
            intent_acc = accuracy_score(list(zip(*batch))[3], intent)
            slot_accs.append(slot_acc)
            intent_accs.append(intent_acc)
        pred_slots_a = np.vstack(pred_slots)
        true_slots_a = np.array(list(zip(*index_test))[2])[:pred_slots_a.shape[0]]

        print("Intent accuracy for epoch {}: {}".format(epoch, np.average(intent_accs)))
        print("Slot accuracy for epoch {}: {}".format(epoch, np.average(slot_accs)))
        print("Slot F1 score for epoch {}: {}.".format(epoch, f1_4_sequence_batch(true_slots_a, pred_slots_a)))

def test_data():
    train_data = open("dataset/atis-2.train.w-intent.iob", "r").readlines()
    test_data = open("dataset/atis-2.dev.w-intent.iob", "r").readlines()
    train_data_ed = data_pre(train_data)
    test_data_ed = data_pre(test_data)
    word2index, index2word, slot2index, index2slot, intent2index, index2intent = \
        get_info_from_training_data(train_data_ed)
        # print("slot2index: ", slot2index)
        # print("index2slot: ", index2slot)
    index_train = to_index(train_data_ed, word2index, slot2index, intent2index)
    index_test = to_index(test_data_ed, word2index, slot2index, intent2index)
    batch = next(getBatch(batch_size, index_test))
    unziped = list(zip(*batch))
    print("word num: ", len(word2index.keys()), "slot num: ", len(slot2index.keys()), "intent num: ",
            len(intent2index.keys()))
    print(np.shape(unziped[0]), np.shape(unziped[1]), np.shape(unziped[2]), np.shape(unziped[3]))
    print(np.transpose(unziped[0], [1, 0]))
    print(unziped[1])
    print(np.shape(list(zip(*index_test))[2]))

if __name__ == '__main__':
    # train(is_debug=True)
    train()
    test_data()