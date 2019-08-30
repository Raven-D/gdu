# encoding=utf-8

import tensorflow as tf
import gdu as g
import sys
import tf_common as tc
import codecs as co

from modelt import Transfer

STAGE = 2

PARAM = {'stage':STAGE,\
         'lr':0.0005 if STAGE == 1 else 0.0005,\
         'l2_rate':1e-4,\
         'emb_size':128 * 4,\
         'hidden_size':128 * 8,\
         'source_vocab_size':10534,\
         'target_vocab_size':10534,\
         'encoder_layers':1,\
         'decoder_layers':2}

STAGE1_DATA = './data/stage1.data'
STAGE2_DATA = './data/stage2.data'
VOCAB = './data/vocab.data'
MODEL_DIR = './models'
TAB = ' ' * 15
TRAIN_STEP = 10000000
BATCH_SIZE = 256 if (PARAM['stage'] == 1) else 128
SAVING_STEP = 10000

def check_value(fname):
    value = 0.0
    try:
        with co.open(fname, 'r', 'utf-8') as rf:
            value = float(rf.read())
    except Exception:
        g.error('NO CONF VALUE FOUND, %s' % fname)
    finally:
        value == value if (None != value) else 0.0
        g.infor('DATA AUGMENT : %f .' % value)
        return value

def train():
    tf.reset_default_graph()
    with tf.Graph().as_default() as global_graph:
        model = Transfer(param=PARAM, mode='train')
        sess_conf = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
        sess_conf.gpu_options.allow_growth = True
        with tf.Session(graph=global_graph, config=sess_conf) as sess:
            model, global_step = tc.load_model(model, MODEL_DIR, sess)
            sess.graph.finalize()
            vocab = g.read_vocab(VOCAB)
            rvocab = g.reverse_vocab(vocab)
            g.infor(TAB + '|begin global step : %d|' % global_step + TAB)
            g.infor(TAB + '|stage-%d|' % PARAM['stage'] + TAB)
            source_label = g.read_contents(STAGE1_DATA if PARAM['stage'] == 1 else STAGE2_DATA, split='=')
            source_label = g.shuffle(source_label)
            source, label = g.unzip_tuple(source_label)
            index = 0
            augment = check_value('./augment')
            # reinit the projection layer before train stage 2.
            # sess.run(model.reinit)
            for i in range(TRAIN_STEP):
                sid, sil, teach, dod, dol, need_shuffle, index = g.get_seq2seq_batch(source=source, label=label,\
                                                  batch_size=BATCH_SIZE, index=index, vocab=vocab, augment=augment)

                _, loss, global_step, lr, tn = model.train(sess, [sid, sil, teach, dod, dol])
                g.normal('step:%d loss:%.4f lr:%.6f TN:%.6f    ' % (global_step, loss, lr, tn), 'fg_green')
                if (need_shuffle):
                    source, label = g.unzip_tuple(g.shuffle(source_label))
                    g.infor('\n' + TAB + '|shuffle data|' + TAB + '\n')
                # save model
                if (need_shuffle):
                    g.record('loss', loss, limit=-1, force_write=True)
                    model.saver.save(sess, MODEL_DIR + '/transfer.ckpt', global_step=global_step)
                    g.infor('                                                                         ')
                    g.infor('                      save model when step to :%d                      ' % global_step)
                    g.infor('                                                                         ')
                    augment = check_value('./augment')
                else:
                    g.record('loss', loss, limit=-1)

TEST = './data/ttest.data'

def infer():
    tf.reset_default_graph()
    with tf.Graph().as_default() as global_graph:
        model = Transfer(param=PARAM, mode='infer')
        sess_conf = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
        sess_conf.gpu_options.allow_growth = True
        with tf.Session(graph=global_graph, config=sess_conf) as sess:
            model, global_step = tc.load_model(model, MODEL_DIR, sess)
            vocab = g.read_vocab(VOCAB)
            rvocab = g.reverse_vocab(vocab)
            g.infor('          |begin global step : %d|          ' % global_step)
            g.infor(TAB + '|stage-%d|' % PARAM['stage'] + TAB)
            source_label = g.read_contents(TEST, split='=')
            source, label = g.unzip_tuple(source_label)
            index = 0
            # save graph pb file
            # tf.train.write_graph(sess.graph_def, MODEL_DIR, 'transfer-infer.pbtxt')
            texts = ''
            CACHE = 512
            infer_len = len(source_label)
            for i in range(TRAIN_STEP):
                sid, sil, teach, dod, dol, need_shuffle, index = g.get_seq2seq_batch(source=source, label=label, batch_size=1, index=index, vocab=vocab)
                if (need_shuffle):
                    break
                sample_id = model.infer(sess, [sid, sil])
                # bs = 1
                sample_id = sample_id[0][0]
                anwser = g.convert_ids_to_string(sample_id, rvocab)
                texts += anwser
                # g.normal('\n')
                # g.normal(g.convert_ids_to_string(sid[0], rvocab))
                # g.normal(anwser)
                # g.normal(g.convert_ids_to_string(dod[0], rvocab))
                # g.normal('\n')
                if (i > 0 and i % CACHE == 0):
                    with co.open('infer-t.txt', 'a', 'utf-8') as wf:
                        wf.write(texts)
                        texts = ''
                    g.infor(TAB + 'progress:%.4f' % (float(i + 1) / infer_len * 100.) + TAB)
            if (texts != ''):
                with co.open('infer-t.txt', 'a', 'utf-8') as wf:
                    wf.write(texts)
                    texts = ''
            g.infor(TAB + 'task infer finished' + TAB)

if __name__ == '__main__':
    argv = sys.argv
    if (len(argv) == 2):
        mode = argv[1]
        g.infor('mode : %s' % mode)
        if (mode == 'infer'):
            g.infor('infer...')
            infer()
        else:
            g.infor('train...')
            train()