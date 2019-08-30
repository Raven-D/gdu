# coding=utf-8
import codecs as co
import os
import sys
import numpy as np
import commands
import time

# valid str , utf8, 20 ~ 40869
CHAR_CODE_BEGIN = 20
CHAR_CODE_END = 40869 + 1

__all__ = ['create_time_tag', 'valid_str', 'normal', 'infor', 'warn', 'error', 'rainbow',\
           'add_line_break', 'shuffle', 'unzip_tuple', 'read_contents', 'tear_to_pieces',\
           'multinomial_read', 'read_vocab', 'reverse_vocab', 'convert_str_to_ids',\
           'convert_ids_to_string', 'padding_array']

join = os.path.join
listdir = os.listdir
exists = os.path.exists
isdir = os.path.isdir
isfile = os.path.isfile

cmd = commands.getstatusoutput

# last element is control sign.
rainbow_seq = [1, 3, 2, 6, 4, 5, 0]

def loop_seq_id(num):
    global rainbow_seq
    if (num == 5):
        rainbow_seq[-1] = 1
    if (num == 0):
        rainbow_seq[-1] = 0
    if (rainbow_seq[-1] == 0):
        return num + 1
    if (rainbow_seq[-1] == 1):
        return num - 1
    return 0

COLOR_END = '\033[0m'
COLOR_END_UN = '\033[4m'
COLORS = {'fg_black': '\033[30m',\
          'fg_red': '\033[31m',\
          'fg_green': '\033[32m',\
          'fg_yellow': '\033[33m',\
          'fg_blue': '\033[34m',\
          'fg_purple': '\033[35m',\
          'fg_cyan': '\033[36m',\
          'fg_white': '\033[37m',\
          'bg_black': '\033[40m',\
          'bg_red': '\033[41m',\
          'bg_green': '\033[42m',\
          'bg_yellow': '\033[43m',\
          'bg_blue': '\033[44m',\
          'bg_purple': '\033[45m',\
          'bg_cyan': '\033[46m',\
          'bg_white': '\033[47m'}

def create_time_tag():
    time_str = time.ctime()
    time_str = time_str[4: (len(time_str) - 5)]
    time_str = '[' + time_str + '] '
    return time_str

def valid_str(info):
    try:
        info = str(info)
    except UnicodeEncodeError:
        # info = info.encode('utf-8')
        return info
    return info

def normal(info, time_tag=False, only_get=False):
    info = valid_str(info)
    if (time_tag):
        info = create_time_tag() + info
    info = COLORS['fg_white'] + u'\u25e5 ' + info + COLOR_END
    if (only_get):
        return info
    print(info)

def infor(info, time_tag=False, only_get=False):
    info = valid_str(info)
    if (time_tag):
        info = create_time_tag() + info
    info = COLORS['fg_white'] + COLORS['bg_blue'] + u'\u263b ' + info + COLOR_END
    if (only_get):
        return info
    print(info)

def warn(info, time_tag=False, only_get=False):
    info = valid_str(info)
    if (time_tag):
        info = create_time_tag() + info
    info = COLORS['fg_black'] + COLORS['bg_yellow'] + u'\u3013' + info + COLOR_END
    if (only_get):
        return info
    print(info)

def error(info, time_tag=False, only_get=False):
    info = valid_str(info)
    if (time_tag):
        info = create_time_tag() + info
    info = COLORS['fg_white'] + COLORS['bg_red'] + u'\u2689 ' + info + COLOR_END
    if (only_get):
        return info
    print(info)

def scolor(info, color='bg_blue', unl=False, time_tag=False, only_get=False):
    info = valid_str(info)
    if (time_tag):
        info = create_time_tag() + info
    info = COLORS[color] + u'\u2663 ' + info + (COLOR_END_UN if unl else COLOR_END)
    if (only_get):
        return info
    print(info)

def lrandom(info , time_tag=False, only_get=False):
    info = valid_str(info)
    if (time_tag):
        info = create_time_tag() + info
    ckeys = COLORS.keys()
    info = COLORS[ckeys[np.random.randint(0, len(ckeys))]] + u'\u273f ' + info + (COLOR_END if np.random.randint(0,2) == 1 else COLOR_END_UN)
    if (only_get):
        return info
    print(info)

rcolorid = 0
def rainbow(info, time_tag=False, only_get=False):
    global rcolorid, rainbow_seq
    info = valid_str(info)
    if (time_tag):
        info = create_time_tag() + info
    rcolorid = loop_seq_id(rcolorid)
    color = '\033[4' + str(rainbow_seq[rcolorid]) + 'm'
    info = color + info + COLOR_END
    if (only_get):
        return info
    print(info)
    

try:
    import matplotlib.pyplot as plt
except ImportError, e:
    warn(e)

def add_line_break(lines):
    '''
    add a line break in the end of every line.
    '''
    if (len(lines) < 1):
        raise ValueError(error('NO VALID LINE FOUND.', time_tag=True, only_get=True))
    for i in range(len(lines)):
        line = lines[i]
        if (not line.endswith('\n')):
            line += '\n'
        lines[i] = line
    return lines

def read_vocab(fname='', fcode='utf-8'):
    '''
    default value: fcode = 'utf-8'
    e.g.: vocab = read_vocab('./vocab.data', fcode='utf-8')
    NOTE: char id is the sequence order in vocab file.
    '''
    vocab = {}
    if (fname == ''):
        return vocab
    with co.open(fname, 'r', fcode) as of:
        lines = of.readlines()
        for line in lines:
            line = line.strip()
            if (vocab.has_key(line)):
                raise ValueError(error('FOUND SAME ITEM IN VOCAB : %s' % line, time_tag=True, only_get=True))
            else:
                vocab[line] = len(vocab)
    return vocab

def reverse_vocab(vocab):
    '''
    reverse vocab from (string, id) to (id, string)
    '''
    rvocab = {}
    for key, value in zip(vocab.keys(), vocab.values()):
        if (rvocab.has_key(value)):
            raise ValueError(error('FOUND SAME ITEM IN REVERSE-VOCAB : %s' % value, time_tag=True, only_get=True))
        else:
            rvocab[value] = key
    return rvocab

# for debug
last = ''
def convert_str_to_ids(string, vocab, unk_id=0, sos=False, sos_id=1, eos=False, eos_id=2):
    '''
    string: the string is a single sentence, not an array, only support utf-8 string.
    vocab: vocab for char-to-id.
    unk_id: <UNK> default is 0.
    sos: whether for adding a <SOS> before string.
    sos_id: <SOS> id in your vocab.
    eos: whether for adding a <EOS> after string.
    eos_id: <EOS> id in your vocab.
    return:
        [1, 4, 55, 354, 444]
        or
        [4, 55, 354, 444, 2]
        ...
    '''
    global last
    sen_arr = []
    string = string.strip()
    if (string.find(' ') > -1):
        string = string.split(' ')
    slen = len(string)
    if (slen < 1):
        infor('LAST SENTENCE: ' + last)
        raise ValueError(error('ENCOUNTER AN EMPTY STRING.', time_tag=True, only_get=True))
    if (sos):
        sen_arr.append(sos_id)
    for st in string:
        if (vocab.has_key(st)):
            sen_arr.append(vocab[st])
        else:
            sen_arr.append(unk_id)
    if (eos):
        sen_arr.append(eos_id)
    last = string
    return sen_arr, len(sen_arr)

def padding_array(arr, padding=0, padding_id=3, padding_type='after', convert_to_numpy=True, dtype=np.int32):
    '''
    ony support 2-d array.
    padding: the length that you want to pad, if 0, we padding array by its max length.
    padding_id: <PAD> id in your vocab.
    padding_type: default is 'after', unless you want to pad the supplyments with type 'before'.
    convert_to_numpy: whether to transfer to numpy array, else will return a python list.
    '''
    max_len = -1
    if (len(arr) < 1):
        raise ValueError(error('EMPTY ERROR IN PADDING.', time_tag=True, only_get=True))
    if (padding == 0):
        for ar in arr:
            if (len(ar) > max_len):
                max_len = len(ar)
    else:
        max_len = padding
    if (max_len == -1):
        raise ValueError(error('CAN NOT GET A VALID MAX LENGTH FOR SUB ERROR.', time_tag=True, only_get=True))
    for i in range(len(arr)):
        ar = arr[i]
        ar = ar[:max_len]
        supp = max_len - len(ar)
        if (supp > 0):
            if (padding_type == 'after'):
                ar += [padding_id] * supp
            else:
                ar = [padding_id] * supp + ar
            arr[i] = ar
        elif (supp < 0):
            raise ValueError(error('PADDING COUNT < 0.', time_tag=True, only_get=True))
    if (convert_to_numpy):
        return np.array(arr, dtype=dtype)
    else:
        return arr

def convert_ids_to_string(ids, rvocab, unk_id=0, unk='?', sos_id=1, eos_id=2, padding_id=3, with_end=True):
    '''
    ids: must be a 1-d array.
    rvocab: is the reverse type of vocab.
    unk_id: <UNK> id in your rvocab.
    unk: what char you want to replace for human reading.
    sos_id: <SOS> id in your rvocab.
    eos_id: <EOS> id in your rvocab.
    padding_id: <PAD> id in your rvocab.
    with_end: determine whether to add \\n at the end of string.
    '''
    sen = u''
    ilen = len(ids)
    if (ilen < 1):
        raise ValueError(error('NO VALID ID FOUND.', time_tag=True, only_get=True))
    for sid in ids:
        if (sid == unk_id):
            sen += unk
        elif (sid == sos_id):
            continue
        elif (sid == eos_id):
            sen += ' <end> '
        elif (sid == padding_id):
            continue
        elif (rvocab.has_key(sid)):
            sen += rvocab[sid]
    if (with_end):
        sen += '\n'
    return sen

def shuffle(arr):
    '''
    support multi-dimension arr, but only take effects on 1st dimension.
    '''
    if (len(arr) < 2):
        warn('NO NECESSARY TO SHUFFLE.')
        return arr
    np.random.shuffle(arr)
    infor('SHUFFLE DATA %d.' % len(arr))
    return arr

jieba = None
try:
    import jieba
except ImportError, e:
    warn(e)

def fabn(x):
    if (len(x) < 1):
        raise ValueError(error('NO ELEMENT FOUND(fabn).', time_tag=True, only_get=True))
    nx = []
    for i in range(len(x)):
        nx.append(sum(x[:i + 1]))
    return nx

def get_section(e, arr):
    if (len(arr) < 1):
        raise ValueError(error('NO ELEMENT FOUND(get_section).', time_tag=True, only_get=True))
    index = 0
    for i in range(len(arr)):
        ai = arr[i]
        if (e <= ai):
            break
        index += 1
    return index


CCHAR_RATE = 0.25
RCHAR_RATE = 0.25
RWORD_RATE = 0.25
RPART_RATE = 0.25
CPART_RATE = 0.50
# e.g. aug_text(u'', 0.5, {'change_char': 0.2, 'reorder_char': 0.2, 'reorder_word': 0.2, 'repeat_part': 0.2, 'clip_part': 0.2})
def aug_text(text, prob, types_and_probs):
    '''
    probs: to control the total probability of augment.
    types_and_probs: a dict to specify the specific type probability.
    e.g.: {'change_char': 0.2, 'reorder_char': 0.2, 'reorder_word': 0.2, 'repeat_part': 0.2, 'clip_part': 0.2}
    '''
    if (prob == 0.0):
        return text
    if (len(text) < 3):
        return text
    if (prob > 1.0):
        raise ValueError(error('PROB SHOULD LESS THAN 1.0 .', time_tag=True, only_get=True))
    if (None == types_and_probs or len(types_and_probs) == 0):
        raise ValueError(error('types_and_probs SHOULD NOT BE NONE.', time_tag=True, only_get=True))
    tp_keys, tp_values = types_and_probs.keys(), types_and_probs.values()
    if (np.sum(tp_values) != 1.0):
        raise ValueError(error('THE SUM OF types_and_probs NOT EQUALS 1.0 .', time_tag=True, only_get=True))
    tp_values = fabn(tp_values)
    # total prob
    nt_arr = []
    if (np.random.randint(1, 101) < 100 * prob):
        type_prob = np.random.randint(1, 101) / 100.
        atype = tp_keys[get_section(type_prob, tp_values)]
        text = text.strip()
        text = list(text)
        tlen = len(text)
        if (atype == 'change_char'):
            vcount = int(tlen * CCHAR_RATE)
            if (vcount > 0):
                vids = np.random.randint(0, tlen, [vcount])
                vnew_chars = np.random.randint(CHAR_CODE_BEGIN, CHAR_CODE_END, [vcount])
                for i in range(vcount):
                    text[vids[i]] = unichr(vnew_chars[i])
        elif (atype == 'reorder_char'):
            vcount = int(tlen * RCHAR_RATE)
            if (vcount > 0):
                vids = np.random.randint(0, tlen, [vcount])
                for i in range(vcount):
                    exid = np.random.randint(0, tlen)
                    tmp = text[exid]
                    text[exid] = text[vids[i]]
                    text[vids[i]] = tmp
        elif (atype == 'reorder_word'):
            text = ''.join(text)
            if (None == jieba):
                return text
            word_ite = jieba.cut(text)
            wa = []
            for e in word_ite:
                wa.append(e)
            tlen = len(wa)
            vcount = int(tlen * RWORD_RATE)
            if (vcount > 0):
                vids = np.random.randint(0, tlen, [vcount])
                for i in range(vcount):
                    exid = np.random.randint(0, tlen)
                    tmp = wa[exid]
                    wa[exid] = wa[vids[i]]
                    wa[vids[i]] = tmp
            text = list(''.join(wa))
        elif (atype == 'repeat_part'):
            if (tlen > 3 * 2):
                slice = np.random.randint(1, tlen-1, [2])
                slice.sort()
                start, end = slice[0], slice[1]
                if (end > start):
                    insertp = end
                    text = text[:start] + text[start:end] * 2 + text[end:]
        elif (atype == 'clip_part'):
            if (tlen > 3 * 2):
                vcount = int(tlen * CPART_RATE)
                if (vcount > 0):
                    text = text[vcount:] if (np.random.randint(1, 3) == 1) else text[:tlen - vcount]
        return ''.join(text)
    return text
                
        
    

def split_list(arr):
    if (len(arr) < 1):
        raise ValueError(error('INVALID LIST.', time_tag=True, only_get=True))
    list1, list2 = [], []
    for item in arr:
        if (len(item) != 2):
            raise ValueError(error('THE ITEM IN LIST LESS THAN 2.', time_tag=True, only_get=True))
        list1.append(item[0])
        list2.append(item[1])
    return list1, list2

def unzip_tuple(tarr):
    '''
    unzip a typle array, e.g.:[(1,2), (3,4) ... ] => [1,3 ...], [2,4 ...]
    return 2 arrays for keys & values
    '''
    if (len(tarr) < 1):
        raise ValueError(error('NO VALID ITEM FOUND.', time_tag=True, only_get=True))
    keys = []
    values = []
    for titem in tarr:
        try:
            k,v = titem
        except ValueError:
            raise ValueError(error('UNZIP ERROR, ' + str(titem), time_tag=True, only_get=True))
        keys.append(k)
        values.append(v)
    return keys, values

def read_contents(cfile='', fcode='utf-8', replace='', split=''):
    '''
    read all contents from file.
    if 'replace' is not empty, the 'replace' will be replaced by ''.
    e.g.: lines = read_contents('./data.txt', replace=' ', split='&')
    NOTE: if you have mass text files to read, please use 'multinomial_read'.
    '''
    if (cfile == ''):
        raise ValueError(error('INVALID SOURCE FILE NAME.', time_tag=True, only_get=True))
    lines = []
    nr = False
    ns = False
    if (replace != ''):
        nr = True
    if (split != ''):
        ns = True
    with co.open(cfile, 'r', fcode) as of:
        lines = of.readlines()
    if (len(lines) > 0):
        for i in range(len(lines)):
            line = lines[i]
            line = line.strip()
            if (nr):
                line = line.replace(replace, '')
            if (ns):
                line = tuple(line.split(split))
            lines[i] = line
    while (lines.count('') > 0):
        lines.remove('')
    return lines

def tear_to_pieces(file_dict=None, cache=1000, teared_path='./teared'):
    '''
    split the huge data file into many small file pieces.
    default: cache=1000
    file_dict e.g.:
    { 'type1': ['type1_file1', 'type1_file2' ... ],
      'type2': ['type2_file1', 'type2_file2' ... ],
      ... }
    for chatting and pre-train liked task, you only have 1 default type,
    name it whatever you like, e.g.: 'chat': ['chat1.txt', 'chat2.txt' ... ]
    '''
    global join, listdir
    if (not exists(teared_path)):
        fd_len = len(file_dict)
        if (fd_len < 1):
            raise ValueError(error('NO VALID FILE DICT FOUND.', time_tag=True, only_get=True))
        infor('found %d types of data.' % fd_len, time_tag=True)
        fd_keys = file_dict.keys()
        # check the validation of type names
        if (len(fd_keys) < 1):
            raise ValueError(error('NO VALID TYPE NAME FOUND.', time_tag=True, only_get=True))
        val_keys = []
        for fk in fd_keys:
            if (len(fk) < 3):
                raise ValueError(error('TYPE NAME MUST LONGER THAN 3.', time_tag=True, only_get=True))
            val_keys.append(fk[:3])
        if (len(list(set(val_keys))) != len(fd_keys)):
            __reason = '''
            THE AHEAD 3 ALPHABETS BETWEEN ALL THE TYPE NAMES MUST BE DIFFERENT.
            WE DO NOT ALLOW THE BELLOWING CONDITION:
            { 'ad': ... # less than 3
              'food_fruite': ...
              'food': ... # the ahead 3 alphabets 'foo' of 3rd item is same with 2nd item 'food_fruite'.
              'common': ...}
            BUT THE BELLOWING IS CORRECT:
            { 'person_name': ...
              'food': ...
              'car': ...
              'places': ...}
            '''
            raise ValueError(error(__reason, time_tag=True, only_get=True))
        # never create teared files
        os.mkdir(teared_path)
        teared_dict_keys = []
        for fdk in fd_keys:
            type_files = file_dict[fdk]
            if (len(type_files) < 1):
                raise ValueError(error('FOUND EMPTY FILE LIST IN TYPE : %s' % fdk, time_tag=True, only_get=True))
            type_files_args = ''
            for tfnames in type_files:
                type_files_args += tfnames
            temp_combined_file = fdk + '-cmb-cat.txt'
            c_s, c_r = cmd('cat ' + type_files_args + ' > ' + temp_combined_file)
            if (c_s != 0):
                raise ValueError(error('CAT FILES ERROR. reason: ' + c_r, time_tag=True, only_get=True))
            teared_file_prefix = fdk + '@@'
            c_s, c_r = cmd('split -l ' + str(cache) + ' ' + temp_combined_file + ' ' + join(teared_path, teared_file_prefix))
            if (c_s != 0):
                raise ValueError(error('SPLIT FILE ERROR. reason: ' + c_r, time_tag=True, only_get=True))
            teared_dict_keys.append(teared_file_prefix)
            c_s, c_r = cmd('rm ' + temp_combined_file)
            if (c_s != 0):
                raise ValueError(error('RM FILE ERROR. reason: ' + c_r, time_tag=True, only_get=True))
    else:
        __ins = '''
        THE TEARED_PATH EXISTED, STEP OVER THIS TASK.
        IF YOU WANT RECREATE TEARED FILES, PLEASE REMOVE THE CURRENT TEARED_PATH FIRST
        AND CALL THIS FUNCTION AGAIN.
        '''
        warn(__ins)
    # construct new file dict
    file_list = listdir(teared_path)
    # exclude dir
    for index in range(len(file_list)):
        flname = file_list[index]
        if (isdir(join(teared_path, flname))):
            file_list[index] = ''
    while (file_list.count('') > 0):
        file_list.remove('')
    if (len(file_list) < 1):
        raise ValueError(error('NO TEARED FILE FOUND.', time_tag=True, only_get=True))
    teared_dict = {}
    for flname in file_list:
        if (flname.find('@@') < 0):
            raise ValueError(error('ERROR TEARED FILE NAME WHICH DOES NOT CONTAIN SPLIT SIGN -> [@@].', time_tag=True, only_get=True))
        type_key, real_name = flname.split('@@')
        if (not teared_dict.has_key(type_key)):
            teared_dict[type_key] = []
        teared_dict[type_key].append(join(teared_path, flname))
    normal('TEARED DICT AS BELLOWING:')
    normal(teared_dict)
    return teared_dict

def __random_softmax__(count):
    nm = np.random.normal(size=count)
    exp = np.exp(nm)
    return exp / np.sum(exp)

def __fetch_nominal_dist__(count, b):
    # gc is file group count
    # b is batch size
    count_len = len(count)
    if (count_len < 1 or b < 1):
        raise ValueError(error('INVALID GROUP COUNT OR BATCH SIZE. [GROUP-COUNT:%d BATCH-SIZE:%d]' % (count_len, b), time_tag=True, only_get=True))
    if (b < count_len):
        b = count_len
        warn('WE RECOMMEND THAT YOUR batch size SHOULD BE BIGGER THAN type count, WE FORCE THE VALUE OF batch size FROM %d TO %d.' % (b, count_len))
    fetch_count = []
    fmeanc = b / count_len
    fetch_count = [fmeanc] * count_len
    mod = b % count_len
    if (mod > 0):
        mod_dist_rate = np.array(count, np.float32) / np.sum(count)
        mod_dist = np.random.multinomial(mod, mod_dist_rate)
        fetch_count += mod_dist
    return fetch_count

def multinomial_read(file_dict=None, cache=1024, fcode='utf-8', replace='', split=''):
    '''
    * using the low time efficiency for exchanging a approximate shuffling process. *
    * we try to emulate a complete random fetch process, this random process can be improved in future days. *
    
    '''
    fd_len = len(file_dict)
    if (fd_len < 1):
        raise ValueError(error('NOT A VALID FILE DICT.', time_tag=True, only_get=True))
    datas = []
    fd_keys, fd_values = file_dict.keys(), file_dict.values()
    file_group_count = []
    for fdv in fd_values:
        file_group_count.append(len(fdv))
    if (len(file_group_count) != fd_len):
        raise ValueError(error('TYPE COUNT NOT EQUALS FILE GROUP COUNT.', time_tag=True, only_get=True))
    type_fetch_count = __fetch_nominal_dist__(file_group_count, cache)
    # print type_fetch_count
    for fdk_index in range(len(fd_keys)):
        fdk = fd_keys[fdk_index]
        fdv = file_dict[fdk]
        sub_data_count = type_fetch_count[fdk_index]
        sub_data_count_dist = np.random.multinomial(sub_data_count, [1. / len(fdv)] * len(fdv))
        # print sub_data_count_dist
        # fetch from every file.
        for ef_index in range(len(sub_data_count_dist)):
            efcount = sub_data_count_dist[ef_index]
            if (efcount == 0):
                continue
            effname = fdv[ef_index]
            eflines = read_contents(effname, fcode=fcode, replace=replace, split=split)
            if (efcount > len(eflines)):
                efcount = len(eflines)
            line_number = np.random.choice(len(eflines), size=efcount, replace=False)
            for ln in line_number:
                datas.append(eflines[ln])
    return datas

def __get_batch__(source, label, batch_size=8, index=0):
    all_len = len(source)
    need_shuffle = False
    start = index
    end = start + batch_size
    if (end > all_len):
        need_shuffle = True
    if (need_shuffle):
        start = (all_len - batch_size)
        end = start + batch_size
        index = 0
    else:
        index += batch_size
    sbatch = source[start:end]
    lbatch = label[start:end]
    return sbatch, lbatch, need_shuffle, index

# Following function is for real task reading.
# Guys you can define your own function by contributing this project.

def get_seq2seq_batch(source=[], label=[], batch_size=8, index=0, vocab=None, unk_id=0, sos_id=1, eos_id=2, dtype=np.int32, augment=0.0, aug_types=None):
    sbatch, lbatch, need_shuffle, index = __get_batch__(source, label, batch_size, index)
    enc_in_data = []
    dec_in_data = []
    dec_out_data = []
    enc_in_len = []
    dec_out_len = []
    if (None == aug_types and augment > 0.0):
        aug_types = {'change_char': 0.2, 'reorder_char': 0.2, 'reorder_word': 0.2, 'repeat_part': 0.2, 'clip_part': 0.2}
    for sen in sbatch:
        sen = aug_text(sen, augment, aug_types)
        _1, _2 = convert_str_to_ids(sen, vocab, unk_id=unk_id, sos=True, sos_id=sos_id)
        enc_in_data.append(_1)
        enc_in_len.append(_2)
    for sen in lbatch:
        _1, _2 = convert_str_to_ids(sen, vocab, unk_id=unk_id, sos=True, sos_id=sos_id)
        dec_in_data.append(_1)
        _1, _2 = convert_str_to_ids(sen, vocab, unk_id=unk_id, eos=True, eos_id=eos_id)
        dec_out_data.append(_1)
        dec_out_len.append(_2)
    enc_in_data = padding_array(enc_in_data, dtype=dtype)
    dec_in_data = padding_array(dec_in_data, dtype=dtype)
    dec_out_data = padding_array(dec_out_data, dtype=dtype)
    enc_in_len = np.array(enc_in_len, dtype=dtype)
    dec_out_len = np.array(dec_out_len, dtype=dtype)
    return enc_in_data, enc_in_len, dec_in_data, dec_out_data, dec_out_len, need_shuffle, index

def get_cls_batch(source=[], label=[], batch_size=8, index=0, vocab=None, fix_padding=0, sos=False, sos_id=1, dtype=np.int32, need_osen=False):
    sbatch, lbatch, need_shuffle, index = __get_batch__(source, label, batch_size, index)
    enc_in_data = []
    enc_in_len = []
    original_sen = []
    for sen in sbatch:
        _1, _2 = convert_str_to_ids(sen, vocab, sos=sos, sos_id=sos_id)
        enc_in_data.append(_1)
        enc_in_len.append(_2)
        original_sen.append(sen)
    label_data = [int(e) for e in lbatch]
    if (fix_padding == 0):
        enc_in_data = padding_array(enc_in_data, dtype=dtype)
    else:
        enc_in_data = padding_array(enc_in_data, padding=fix_padding, dtype=dtype)
    enc_in_data = np.array(enc_in_data, dtype=dtype)
    enc_in_len = np.array(enc_in_len, dtype=dtype)
    label_data = np.array(label_data, dtype=dtype)
    if (need_osen):
        return enc_in_data, enc_in_len, label_data, need_shuffle, index, original_sen
    else:
        del original_sen
        return enc_in_data, enc_in_len, label_data, need_shuffle, index

def writef(fname='', value='', mode='a', fcode='utf-8'):
    if (fname == '' or value == ''):
        raise ValueError(error('INVALID FILENAME OR VALUE.', time_tag=True, only_get=True))
    with co.open(fname, mode, fcode) as wf:
        # value = valid_str(value)
        if (not value.endswith('\n')):
            value += '\n'
        wf.write(value)
        wf.flush()

records = {}
def record(key='', value=np.inf, limit=1000, force_write=False):
    global records
    if (key == '' or value == np.inf):
        raise ValueError(error('INVALID KEY OR VALUE IN RECORD.', time_tag=True, only_get=True))
    if (not records.has_key(key)):
        records[key] = []
    else:
        records[key].append(value)
    if (len(records[key]) == limit or force_write):
        mean = np.mean(records[key])
        writef(key, valid_str(mean))
        records[key] = []

def draw_diagram(key='', color='k-', lw=2):
    if (key == ''):
        raise ValueError(error('INVALID KEY WHEN DRAW DIAGRAM.', time_tag=True, only_get=True))
    try:
        datas = co.open(key, 'r', 'utf-8').readlines()
        ndatas = []
        for d in datas:
            d = d.strip()
            if (d != ''):
                try:
                    ndatas.append(float(d))
                except ValueError:
                    raise ValueError(error('INVALID FLOAT NUMBER.', time_tag=True, only_get=True))
        del datas
        if (len(ndatas) != 0):
            plt.figure(figsize=(22,12))
            plt.plot(range(1, len(ndatas) + 1), ndatas, color, lw=lw)
            plt.show()
    except IOError:
        raise ValueError(error('NO VALID FILE FOUND (%s).' % key, time_tag=True, only_get=True))

def test_case():
    infor('TEST CASE')
    # read vocab
    vocab = read_vocab('./data/vocab.data')
    rvocab = reverse_vocab(vocab)
    # test huge data size
    # newd = tear_to_pieces({'application':['./data/appfinal.txt'], 'tvchannel':['./data/channelfinal.txt'], 'personchat':['./data/chatfinal.txt'],\
    #                        'converter':['./data/converterfinal.txt'], 'couplets':['./data/coupletfinal.txt'], 'disport':['./data/disportfinal.txt']}, cache=200)
    # for i in range(10):
    #     source_label = multinomial_read(newd, cache=190, replace=' ', split='=')
    #     source, label = unzip_tuple(source_label)
    #     index = 0
    #     need_shuffle = False
    #     while (need_shuffle == False):
    #         eid, eil, ld, need_shuffle, index = get_cls_batch(source=source, label=label, batch_size=8, index=index, vocab=vocab, fix_padding=40)
        
    # test small data size
    source_label = read_contents('./data/chat.txt', replace=' ', split='=')
    source, label = unzip_tuple(source_label)
    index = 0
    for i in range(500):
        eid, eil, did, dod, dol, need_shuffle, index = get_seq2seq_batch(source, label, 256, index, vocab)
        normal('shuffle:%r , index:%d' % (need_shuffle, index))
        if (need_shuffle):
            source, label = unzip_tuple(shuffle(source_label))

if __name__ == '__main__':
    argv = sys.argv
    if (len(argv) == 2):
        diagram = argv[1]
        draw_diagram(diagram)
