# GDU Instruction

### GDU what we create is plan to be used for handle huge data size, esp for sequence data.
### For huge data size, we emulate a multinomial distribution to fetch all kinds of data, by using this emulation we would get an approximate global shuffling process.
### So if the current emulation method is not good for task, you could make a better sulotion by yourself own to improve this module.

#### Here's some API instruction as following:

## PREPARATION

##### We recommand that you should attach the 'source' data and 'labels' data together.
##### For doing this, you could use linux command 'paste'.
##### If you hold 2 files, one is full of "input sentence" and another is full of "reply sentence".
##### You need to combine the 2 files into 1 file with a sign '=', ofcourse you can choose another split sign and be care that it would be better what your text data does not contain the split sign.

```
paste -d = input.data replay.data > chat.data
```
###### By then, you data will looks like this:

```
很好行认识你=我也是
say hello=hello!
Chao!=Chao bella!
...
...
```

### Specifically, we show some exmaple data as following:

**TASK CLASSIFICATION**


```
# positive data type id = 0
# negative data type id = 1

你好很高兴见到你=0
我不怎么喜欢你=1
Glad to see u guys!=0
What the hell did you do?!=1
...
...
```

**TASK CHIT-CHAT**


```
你好啊=你也好！
我很好=我还行！
Good to see you=me, too
Farewell=Dont say that!
...
...
```

**TASK SEQUENCE TAGGING**

###### Because the standard sequence tagging sign is not a single char, we recommand that you would better translate your own tag sign to a single char.
###### e.g.: PLACE_B => a, PLACE_E => b, NAME_B => c, ...

```
请观看今天的人民日报=bababbaaab
下午3点去吃饭=ababbab
...
...
```

## HOW TO USE


```
import gdu as g

# init your vocab
vocab = read_vocab('./data/vocab.data')
# make your reverse vocab to convert ids to chars
rvocab = reverse_vocab(vocab)

# for huge data size
# here cache is the file lines for file pieces
newd = g.tear_to_pieces({'application':['./data/appfinal.txt'], 'tvchannel':['./data/channelfinal.txt'], 'personchat':['./data/chatfinal.txt'], 'converter':['./data/converterfinal.txt'], 'couplets':['./data/coupletfinal.txt'], 'disport':['./data/disportfinal.txt']}, cache=2048)
# define your own TRAIN_STEP
for i in range(TRAIN_STEP):
    # here cache is how many datas you want to read from all files, we recommend you choose a big integer but less that file piece.
    source_label = multinomial_read(newd, cache=1024, replace=' ', split='=')
    source, label = unzip_tuple(source_label)
    index = 0
    need_shuffle = False
    while (need_shuffle == False):
        eid, eil, ld, need_shuffle, index = get_cls_batch(source=source, label=label, batch_size=8, index=index, vocab=vocab, fix_padding=40)

# for small data size, you could read all of data in RAM.
source_label = read_contents('./data/chat.txt', replace=' ', split='=')
source, label = unzip_tuple(source_label)
index = 0
for i in range(110 * 3):
    # eid is encoder input data
    # eil is encoder input data length
    # did is decoder input data
    # dod is decoder output data
    # dol is decoder output data length
    eid, eil, did, dod, dol, need_shuffle, index = get_seq2seq_batch(source, label, 256, index, vocab)
    rainbow('shuffle:%r , index:%d' % (need_shuffle, index))
    if (need_shuffle):
        source, label = unzip_tuple(shuffle(source_label))
```

##### We also have some colorful logging tools for you:
normal, infor, warn, error and **rainbow**

If you choose the logging 'rainbow' it will looks like as follow:

![image](http://wx1.sinaimg.cn/large/65bb9c9cly1g5d2w5869vj21fq0faanu.jpg)

### LUCKY FOR YOU.

----

### tc_common is a module for conveniently ref tensorflow base functions.

##### It will includes more and more high ensemble 'prevailing block', hold expectations pls.
