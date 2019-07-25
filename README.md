# GDU Instruction

### GDU what we create is plan to be used for handle huge data size, esp for sequence data.
### For huge data size, we emulate a multinomial distribution to fetch all kinds of data, by using this emulation we would get an approximate global shuffling process.
### So if the current emulation method is not good for task, you could make a better sulotion by yourself own to improve this module.

#### Here's some API instruction as following:

## PREPARATION

##### We recommand that you should attach the 'source' data and 'labels' data together.
##### For doing this, you could use linux command 'paste'.
##### e.g.: TASK: chatting
##### e.g.: 'source' file: source.data; 'label' file: label.data
##### You can use '=' the equal sign to combine source and label.

```
paste -d = source.data label.data > chat.data
```
###### By then, you data will looks like this:

```
Whats your name?=I have no name, you can call me whatever you will.
What are you talking about?=You guess what? I let the dogs out!
...
...
```


---

##### For tasks:


- chat - | hello John!=Hello Anna! |
- classification - | may I order a pack of cigarrete?=3 | 3 is the class id.
- language model - | who are you=who are you | source is same with label.
- ...


##### You also should prepare your vocab data in advance. e.g.: vocab.data

##### 


##### For reading data in:


```
import gdu as g

source_label = g.read_contents(cfile='chat.data', fcode='utf-8', replace=' ', split='=')
source_label = g.shuffle(source_label)
vocab = g.read_vocab('vocab.data', fcode='utf-8')
rvocab = g.reverse_vocab(vocab)
source, label = g.unzip_tuple(source_label)
...
# enter loop of training
sbatch, lbatch, need_shuffle = g.get_batch(source, label, batch_size=8, count=global_step)
if (need_shuffle):
    source, label = g.unzip(g.shuffle(zip(source, label)))
...
do your training
...
```
