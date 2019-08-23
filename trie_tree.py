# -*- coding: utf-8 -*-

import json


class TrieTree:
    def __init__(self, init_tree_path=None, tree_save_path='./trie_tree.json'):
        """
        :param init_tree_path: 初始化trie_tree文件
        :param tree_save_path: 保存trie_tree路径
        """
        if init_tree_path:
            self.trie_tree = self.load_tree(init_tree_path)
        else:
            self.trie_tree = {}

        self.tree_save_path = tree_save_path

    def load_tree(self, init_tree_path):
        """加载文件"""
        with open(init_tree_path) as f:
            return json.loads(f.read())

    def insert(self, list_str):
        """向trie_tree插入数据"""
        tree = self.trie_tree
        for word in list_str:
            word = word.lower()
            for char in word:
                if char in tree:
                    tree = tree[char]
                else:
                    tree[char] = {}
                    tree = tree[char]
            tree['word'] = word
            tree = self.trie_tree

    def search(self, query, type=0):
        """
        搜索词语是否存在tree中
        :param query: 查询词
        :param type: 类别：
                       0：仅查询； 1：若不存在，在tree中添加
        :return:
        """
        if type == 0:
            tree = self.trie_tree
            query = query.lower()
            for char in query:
                if char in tree:
                    tree = tree[char]
                else:
                    return "{} not in tree !".format(query)
            if "word" in tree:
                return "{} in tree !".format(query)
        else:
            flag = False
            tree = self.trie_tree
            query = query.lower()
            for char in query:
                if char in tree:
                    tree = tree[char]
                else:
                    flag = True
                    tree[char] = {}
                    tree = tree[char]
            if "word" in tree:
                return "{} in tree !".format(query)
            tree['word'] = query
            if flag:
                return "{} not in tree !".format(query)

    def delete(self, query):
        """删除trie_tree中节点"""
        tree = self.trie_tree
        query = query.lower()
        for char in query:
            if char in tree:
                tree = tree[char]
            else:
                return "{} not in tree !".format(query)
            if "word" in tree:
                tree.pop("word")
                return "{} delete !".format(query)

    def save(self):
        """保存trie_tree"""
        with open(self.tree_save_path, 'w', encoding='utf-8') as f:
            json.dump(self.trie_tree, f, ensure_ascii=False, indent=4)
            print("Trie tree saved to {}".format(self.tree_save_path))


if __name__ == '__main__':
    tree = TrieTree()
    tree.insert(['中国', '美国', '中美'])
    print(tree.search('中国'))
    print(tree.delete('中美'))
    print(tree.trie_tree)
    tree.save()