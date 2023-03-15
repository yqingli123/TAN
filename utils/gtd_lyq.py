# coding=utf-8
import pickle


def gtd2latex(gtd_list):
    structures = ['⿶', '⿲', '⿴', '⿵','⿷', '⿺', '⿳', '⿹', '⿱', '⿻', '⿸', '⿰']

    # 统计 parent_idx list
    parent_idx_list = []
    for gtd in gtd_list:
        parent_idx = gtd[3]
        parent_idx_list.append(parent_idx)
    
    # 判断每个子节点是否当过 parent
    parent_times = dict()    # 记录每个 parent_node 当前是第几次当父节点
    caption_list = []
    caption_idx_list = []
    for i, gtd in enumerate(gtd_list):
        if i > 0 :
            child_char = gtd[0]
            child_idx = gtd[1]
            parent_char = gtd[2]
            parent_idx = gtd[3]

            if parent_idx not in list(parent_times.keys()):
                parent_times[parent_idx] = 0
            else:
                parent_times[parent_idx] += 1
            
            # 处理 parent_node
            if parent_times[parent_idx] == 0 :
                caption_list.append(parent_char)
                caption_idx_list.append(parent_idx)
             

            # 处理 child_node  。。。。。 不确定是否就这一种情况 。。。。。。
            if child_idx not in parent_idx_list:
                caption_list.append(child_char)
                caption_idx_list.append(child_idx)
    
    # 后处理 去掉 <eol> End
    try:
        caption_list.remove('<eol>')
        caption_list.remove('End')
    except:
        x = 1

    return caption_list

if __name__ == '__main__':

    char2tree_pkl = '../char2newtree_v1.pkl'
    with open(char2tree_pkl, 'rb') as f:
        char2tree_dict = pickle.load(f)
    
    char2IDS_txt = '../../char2IDS_7144.txt'
    char2IDS_dict = dict()
    f = open(char2IDS_txt, 'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        char = line[0]
        IDS_str = line[2:]
        IDS_list = IDS_str.strip().split()
        char2IDS_dict[char] = IDS_list

    num = 0
    for i, (char, tree_gtd) in enumerate(char2tree_dict.items()):
        caption_list = gtd2latex(tree_gtd)
        print(caption_list)
        groundtruth = char2IDS_dict[char]
        if groundtruth != caption_list:
            num += 1
            print(char)
            print(tree_gtd)
            print(caption_list)
    print(num)