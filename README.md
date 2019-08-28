# GDU & TF_COMMON Instructions
* **gdu.py is a convenient functions tool for data processing.**
* **tf_common.py is a high level and convenient wrapper for TensorFlow APIs, include some preveiling blocks.**

----

For using **GDU** module, you should follow these steps:

data format:
```
> sequence_source=sequence_label
> e.g.
>   for CHAT, 你好=你也好啊！
>   for CLS, 新华社消息……=2 # (2 is the id of news domain)
>   for TAG, 中华人民站起来了=BEBEBIEE
```

For import module:
```
import gdu as g
```

For reading vocab:
```
vocab = g.read_vocab(VOCAB_PATH)
reverse_vocab = g.reverse_vacab(vocab)
```

If your data can be fully loaded in RAM, you could load data as the following:
```
source_label = g.read_contents(FILE_PATH, replace=' ', split='=')
source, label = g.unzip_tuple(source_label)
index = 0
for i in range(TRAIN_STEP):
    eid, eil, did, dod, dol, need_shuffle, index = g.get_seq2seq_batch(source, label, 256, index, vocab)
    if (need_shuffle):
        source, label = g.unzip_tuple(g.shuffle(source_label))
```

If your data is too huge and can not be fully loaded into RAM, you can use **{multinomial_read}** to simulate a global shuffle.
```
newd = g.tear_to_pieces({'class1':['./data/class1.txt'],\
                         'class2':['./data/class2.txt'],\
                         'class3':['./data/class3.txt'],\
                         'class4':['./data/class4.txt'],\ 
                         'class5':['./data/class5.txt'],\ 
                         'class6':['./data/class6.txt']},\
                         cache=204800)
for i in range(TRAIN_STEP):
    source_label = multinomial_read(newd, cache=1024, replace=' ', split='=')
    source, label = unzip_tuple(source_label)
    index = 0
    need_shuffle = False
    while (need_shuffle == False):
        eid, eil, ld, need_shuffle, index = get_cls_batch(source=source, label=label, batch_size=8, index=index, vocab=vocab, fix_padding=40)
```

----

In **tf_common** module, we build lots of convenient tensorflow APIs and some preveiling NN blocks.
e.g.:
- *bahdanau_att, create bahdanau attention wrapper class.*
- *dselfatt, createt self-attention for dynamic sequence length.*
- *sselfatt, create self-attention for static sequence length.*
- *bi_gru, create a bi-direction GRU cell list.*
- *gelu & ngelu, gelu activation function from BERT and the one with normed operation.*
- *...*
- *...*

You can use `tf_common.flist()` to show all the functions and its' helping documents.

----

Additionally, we have some lovely LOG functions for beautifying you command-line.
- *gdu.normal*
- *gdu.infor*
- *gdu.warn*
- *gdu.error*

And some more fantastic LOG functions.
- *gdu.scolor*
- *gdu.lrandom*
- *gdu.rainbow*

Some of them will output information like the following:

![LOG function](https://github.com/Raven-D/gdu/blob/master/assets/log.png?raw=true "LOG function")

----
#### Enjoy it and for good luck.