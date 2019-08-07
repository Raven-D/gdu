# encoding=utf-8
import sys
import os
import gdu as g
from model import Cls
import tf_common as tc

tf = tc.tf

PARAM = {'lr': 0.0005,\
         'lr_drate': 0.9925,\
         'lr_dstep':1000,\
         'lr_dlimit':0.0001,\
         'l2_rate': 1e-3,\
         'emb_size': 128,\
         'source_vocab_size':10511,\
         'gru_layers':2,\
         'cls_num':27}

MODEL_DIR = './models'
VOCAB = './data/vocab.data'
DATA = './data/cls.data'
TRAIN_STEP = 1000000
BATCH_SIZE = 27 * 2
SAVING_STEP = 10000

def train():
    # begin graph
    tf.reset_default_graph()
    with tf.Graph().as_default() as global_graph:
        model = Cls(param=PARAM, mode='train')
        # make tf config
        sess_conf = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
        sess_conf.gpu_options.allow_growth = True
        with tf.Session(graph=global_graph, config=sess_conf) as sess:
            model, global_step = tc.load_model(model, MODEL_DIR, sess)
            sess.graph.finalize()
            g.rainbow('          |BEGIN GLOBAL STEP : %d|          ' % global_step)
            # prepare data
            vocab = g.read_vocab(VOCAB)
            rvicab = g.reverse_vocab(vocab)
            source_label = g.read_contents(DATA, split='=')
            source_label = g.shuffle(source_label)
            source, label = g.unzip_tuple(source_label)
            index = 0
            for i in range(TRAIN_STEP):
                eid, eil, ld, need_shuffle, index = g.get_cls_batch(source=source, label=label, batch_size=BATCH_SIZE, index=index, vocab=vocab)
                if (need_shuffle):
                    source, label = g.unzip_tuple(g.shuffle(source_label))
                    g.rainbow('          |SHUFFLE DATA|          ')
                _, ce_loss, l2_loss, global_step, lr = model.train(sess, [eid, eil, ld])
                g.rainbow('CE_LOSS:%.4f, L2LOSS:%.4f STEP:%d (LR:%.6f)             ' % (ce_loss, l2_loss, global_step, lr))
                g.record('CELOSS', ce_loss)
                g.record('L2LOSS', l2_loss)
                g.record('TLOSS', (ce_loss + l2_loss) / 2.0)
                # save model
                if (global_step > 0 and global_step % SAVING_STEP == 0):
                    model.saver.save(sess, MODEL_DIR + '/cls.ckpt', global_step=global_step)
                    g.warn('                                                                         ')
                    g.warn('                      SAVE MODEL WHEN STEP TO :%d                      ' % global_step)
                    g.warn('                                                                         ')


TEST = './data/test.data'
def infer():
    with tf.Graph().as_default() as global_graph:
        model = Cls(param=PARAM, mode='train')
        with tf.Session(graph=global_graph) as sess:
            model, global_step = tc.load_model(model, MODEL_DIR, sess)
            vocab = g.read_vocab(VOCAB)
            source_label = g.read_contents(TEST, split='=')
            source, label = g.unzip_tuple(source_label)
            index = 0
            acc = 0.0
            err = 0.0
            for i in range(len(source_label)):
                eid, eil, ld, need_shuffle, index, osen = g.get_cls_batch(source=source, label=label, batch_size=1, index=index, vocab=vocab, need_osen=True)
                res = model.infer(sess, [eid, eil])
                if (res[0][0] == ld[0]):
                    acc += 1.
                else:
                    evalue = osen[0] + ':' + str(res[0][0]) + ' / ' + str(ld[0])
                    g.writef('evalue', evalue)
                    err += 1
                if (need_shuffle):
                    break;
                if (i > 0 and i % 100 == 0):
                    prog_percent = (i + 1) / float(len(source_label)) * 100.
                    acp = int(err / prog_percent)
                    g.scolor('  =>  PROGRESS:%.4f [All count prediction: %d] ' % (prog_percent, acp), color='bg_green')
            g.rainbow('INDEX:%d ACC:%.4f' % (index, acc / float(len(source_label))))

if __name__ == '__main__':
    argv = sys.argv
    mode = 'train'
    if (len(argv) == 2):
        mode = str(argv[1])
    g.rainbow('MODE: ' + mode)
    if (mode == 'train'):
        train()
    else:
        infer()
