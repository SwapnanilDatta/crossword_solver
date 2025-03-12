class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search_pattern(self, pattern, node=None, prefix=""):
        if node is None:
            node = self.root

        if not pattern:
            return [prefix] if node.is_end_of_word else []

        results = []
        char = pattern[0]

        if char == "_":
            for key, child_node in node.children.items():
                results.extend(self.search_pattern(pattern[1:], child_node, prefix + key))
        elif char in node.children:
            results.extend(self.search_pattern(pattern[1:], node.children[char], prefix + char))

        return results
