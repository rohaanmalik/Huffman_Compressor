"""
Code for compressing and decompressing using Huffman compression.
"""

from assignments.a2.starter.nodes import HuffmanNode, ReadNode

# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    >>> byte_to_bits(1)
    '00000001'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict{int,int}

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    result = {}
    for i in text:
        if i not in result:
            result[i] = 1
        else:
            result[i] += 1
    return result


def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4, 5: 5, 6: 6}
    >>> t = huffman_tree({0: 1, 1: 1})
    >>> t == HuffmanNode(None, HuffmanNode(0, None, None), HuffmanNode(1, None,\
     None))
    True
    """

    items = list(freq_dict.items())
    items.sort(key=lambda x: x[1])
    data = items

    node_list = []
    for i in data:
        node = HuffmanNode(i[0])
        node.number = i[1]
        node_list.append(node)

    while len(node_list) != 1:
        # Keeps track of order of tuples
        a, b = node_list.pop(0), node_list.pop(0)

        if a.number <= b.number:
            root = HuffmanNode(None, a, b)
        else:
            root = HuffmanNode(None, b, a)

        root.number = a.number + b.number
        node_list.append(root)
        node_list.sort(key=lambda x: x.number)

    return node_list[0]


def get_codes(tree):
    """ Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree =  HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    data = pre_orderhf(tree, {})
    if None in data:
        del data[None]
    return data


def pre_orderhf(t, result, symbol=''):
    """
    Trial
    """
    if t is not None:
        # reslt = result.copy() if result else {}
        # if t.symbol != None and t.is_leaf()
        result[t.symbol] = symbol
        # print(t.symbol, symbol)
        pre_orderhf(t.left, result, symbol=symbol+'0')
        # print(t.symbol, symbol)
        pre_orderhf(t.right, result, symbol=symbol+'1')
        # print(t.symbol, symbol)
    return result


def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, HuffmanNode(0, None, None), HuffmanNode(1, \
    None, None))
    >>> number_nodes(tree)
    >>> tree.number
    0
    """
    postorder(tree, count=[0])


def postorder(tree, count):
    """
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> t = HuffmanNode(None, HuffmanNode(6, None, None), HuffmanNode(None, \
    HuffmanNode(None, HuffmanNode(3, None, None), HuffmanNode(None, \
    HuffmanNode(1, None,
    None), HuffmanNode(2, None, None))), HuffmanNode(None, HuffmanNode(4, None,\
     None), HuffmanNode(5, None, None))))
    >>> t = HuffmanNode(None, HuffmanNode(0, None, None), HuffmanNode(1, None,\
     None))
    >>> postorder(t)
    >>> t.number
    >>> 0
    """
    # count = count.copy() if count else [0]
    if tree is not None:
        # count = count.copy() if count else [0]
        postorder(tree.left, count)
        postorder(tree.right, count)
        # count = count.copy() if count else [0]
        # print(tree.symbol, count)
        if tree.symbol is None:
            # count = count.copy() if count else [0]
            tree.number = count[-1]
            count.append(count[-1] + 1)


def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    all_codes = get_codes(tree)
    sum_freq = 0
    for key in freq_dict:
        sum_freq += freq_dict[key]
    assert sum_freq != 0
    total_chars = 0
    if len(all_codes) > 0:
        # To check if dictionary is empty
        for key in all_codes:
            # Multipliyng length of codes by \
            # frequency to get toal number of chars
            if key in all_codes and key in freq_dict:
                total_chars += len(all_codes[key]) * freq_dict[key]
        return total_chars/sum_freq
    else:
        return 0.0


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mappings from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    >>> text = bytes([65, 66, 67, 66])
    >>> codes = {65: '10', 66: '0', 67: '11'}
    >>> generate_compressed(text, codes)
    r
    """
    byte_list = []
    codes_ = ''
    for item in text:
        codes_ += codes[item]
    # return codes_
    req = len(codes_) // 8
    remainder = len(codes_) - req * 8
    for i in range(req):
        # byte_list.append(codes_[:8])
        a = codes_[8 * i: 8 * i + 8]
        byte_list.append(a)
    if len(codes_) % 8 != 0:
        remainder_code = codes_[req * 8:]
        # byte_list.append(codes_[req*)
        remainder_code += '0' * (8 - remainder)
        byte_list.append(remainder_code)

    list1 = []
    for item in byte_list:
        list1.append(bits_to_byte(item))
    if len(list1) > 0:
        ibyte = bytes([list1[0]])
        # return ibyte
        for i in range(1, len(list1)):
            ibyte += bytes([list1[i]])
        return ibyte
    else:
        return bytes([])


def tree_to_bytes(tree):
    """ Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    return helper(tree, ibytes=[])


def helper(tree, ibytes):
    """
    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(helper(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(helper(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    if tree is not None:
        helper(tree.left, ibytes)
        helper(tree.right, ibytes)
        # print(tree.symbol, ibytes)
        if tree.symbol is None:
            if tree.left.is_leaf():
                ibytes.append(0)
            if not tree.left.is_leaf():
                ibytes.append(1)
            if tree.left.is_leaf():
                ibytes.append(tree.left.symbol)
            if not tree.left.is_leaf():
                ibytes.append(tree.left.number)
            if tree.right.is_leaf():
                ibytes.append(0)
            if not tree.right.is_leaf():
                ibytes.append(1)
            if tree.right.is_leaf():
                ibytes.append(tree.right.symbol)
            if not tree.right.is_leaf():
                ibytes.append(tree.right.number)
    return bytes(ibytes)


def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer that we want to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file whose contents we want to compress
    @param str out_file: output file, where we store our compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
    HuffmanNode(12, None, None)), HuffmanNode(None, HuffmanNode(5, None, None),\
     HuffmanNode(7, None, None)))
    >>> lst2 = [ReadNode(1, 1, 1, 3), ReadNode(0, 10, 1, 2), \
    ReadNode(0, 5, 0, 4), ReadNode(0, 15, 0, 7)]
    >>> generate_tree_general(lst2, 0)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
    HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(4, None, None))),\
     HuffmanNode(None, HuffmanNode(15, None, None), HuffmanNode(7, None, None)))
    >>> lst = [ReadNode(1, 1, 1, 2), ReadNode(0, 10, 0, 12), \
    ReadNode(0, 5, 0, 7)]
    >>> generate_tree_general(lst, 0)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
    HuffmanNode(12, None, None)), HuffmanNode(None, HuffmanNode(5, None, None),\
     HuffmanNode(7, None, None)))
    >>> lst3 = [ReadNode(0, 1, 0, 2)]
    >>> generate_tree_general(lst3, 0)
    HuffmanNode(None, HuffmanNode(1, None, None), HuffmanNode(2, None, None))
    >>> lst4 = [ReadNode(1, 1, 1, 3), ReadNode(0, 3, 1, 2), \
    ReadNode(0, 7, 0, 8), ReadNode(0, 5, 0, 6)]
    >>> generate_tree_general(lst4, 0)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(3, None, None), \
    HuffmanNode(None, HuffmanNode(7, None, None), HuffmanNode(8, None, None))),\
     HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(6, None, None)))
    >>> lst5 = [ReadNode(1, 3, 1, 1), ReadNode(1, 2, 0, 3), \
    ReadNode(0, 7, 0, 8), ReadNode(0, 5, 0, 6)]
    >>> generate_tree_general(lst5, 0)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
    HuffmanNode(6, None, None)), HuffmanNode(None, HuffmanNode(None, \
    HuffmanNode(7, None, None), HuffmanNode(8, None, None)), \
    HuffmanNode(3, None, None)))
    >>> lst6 = [ReadNode(1, 1, 1, 4), ReadNode(1, 2, 1, 3), \
    ReadNode(0, 1, 0, 2), ReadNode(0, 3, 0, 4), ReadNode(0, 5, 0, 6)]
    >>> generate_tree_general(lst6, 0)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(None, \
    HuffmanNode(1, None, None), HuffmanNode(2, None, None)), \
    HuffmanNode(None, HuffmanNode(3, None, None), HuffmanNode(4, None, None))),\
     HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(6, None, None)))
    """
    return general_helper(HuffmanNode(None, HuffmanNode(None), HuffmanNode(None)),
                     node_lst, node_lst[root_index])


def general_helper(node, node_lst, root):
    """ Return a tree of Huffman node from the node list and
    the node that takes in with the root.

    @param HuffmanNode node: a huffman node
    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param ReadNode root: a read node.
    @rtype: HuffmanNode
    """

    if node.left is None:
        node.left = HuffmanNode(None)
    if root.l_type == 0:
        # print('reached left 0', node)
        node.left = HuffmanNode(root.l_data)
        # print('hua')
    if node.right is None:
        node.right = HuffmanNode(None)
    if root.r_type == 0:
        node.right = HuffmanNode(root.r_data)
        # print('reached right 0', node)
    # print('reached else', node)
    if root.l_type == 1:
        node_left = node_lst[root.l_data]
        general_helper(node.left, node_lst, node_left)
    if root.r_type == 1:
        node_right = node_lst[root.r_data]
        general_helper(node.right, node_lst, node_right)
    return node


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that the list represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
    HuffmanNode(7, None, None)), HuffmanNode(None, HuffmanNode(10, None, None),\
     HuffmanNode(12, None, None)))
    >>> lst1 = [ReadNode(0,1,0,2),ReadNode(0,3,0,4), ReadNode(1,8,1,9)]
    >>> generate_tree_postorder(lst1, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(1, None, None), \
    HuffmanNode(2, None, None)), HuffmanNode(None, HuffmanNode(3, None, None),\
    HuffmanNode(4, None, None)))
    >>> lst2 = [ReadNode(0,1,0,2),ReadNode(1,5,0,3), \
    ReadNode(0,4,0,5),ReadNode(1,7,1,8)]
    >>> generate_tree_postorder(lst2, 3)
    True
    """
    return helper2(node_lst, node_lst[root_index],
                   HuffmanNode(None, HuffmanNode(None), HuffmanNode(None)))


def helper2(lst, root, node):
    """
    helper for postorder
    first right, then left!
    """

    if root.r_type == 0:
        # for leaf
        if isinstance(node, HuffmanNode):
            node.right = HuffmanNode(root.r_data)
    else:
        if len(lst) >= 2:
            # if isinstance(node, HuffmanNode):
            helper2(lst, lst[-2], node.right)
    if root.l_type == 0:
        if isinstance(node, HuffmanNode):
            node.left = HuffmanNode(root.l_data)
            lst.remove(root)
    else:
        if len(lst) >= 2:
            if isinstance(node, HuffmanNode):
            # if node.left is None and node.right is not None:
            #     node.left = HuffmanNode(None)
                helper2(lst, lst[-2], node.left)


    return node

#
# lst2 = [ReadNode(0,1,0,2),ReadNode(1,5,0,3), \
# ReadNode(0,4,0,5),ReadNode(1,7,1,8)]
# lst = [ReadNode(0,1,0,2),ReadNode(0,3,0,4), ReadNode(1,8,1,9)]
# generate_tree_postorder(lst, 2)


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes
    >>> t = HuffmanNode(None, HuffmanNode(None, HuffmanNode(3), \
    HuffmanNode(None, HuffmanNode(1), HuffmanNode(4))), \
    HuffmanNode(None, HuffmanNode(2), HuffmanNode(5)))
    >>> text = bytes([216, 0])
    >>> size = 4
    >>> a =  bytes([5, 4, 3, 3])
    >>> a == generate_uncompressed(t, text, size)
    True
    """
    bits = []
    for byte in text:
        bits.append(byte_to_bits(byte))

    evaluate = ''
    for i in bits:
        evaluate += i

    codes = get_codes(tree)
    inv_code = {v: k for k, v in codes.items()}
    i = 0
    j = 1
    result = []
    to_find = evaluate[i:j]
    while len(result) < size:
        if to_find in inv_code:
            result.append(inv_code[to_find])
            j = j + 1
            i = j - 1
            to_find = evaluate[i: j]
        else:
            i = i
            j = j + 1
            to_find = evaluate[i: j]

    return bytes(result)


#
# tree = HuffmanNode(None, HuffmanNode(66, None, None), \
# HuffmanNode(None, HuffmanNode(65, None, None), HuffmanNode(67, None, None)))
# text = b'\x98'
# print(generate_uncompressed(tree, text, 4))

def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType
    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {98: 23, 97: 26, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    >>> freq2 = {100: 15, 101: 17, 98: 20, 103: 21, 105: 23}
    >>> tree2 = HuffmanNode(None, HuffmanNode(None, HuffmanNode(100),   \
    HuffmanNode(None, HuffmanNode(98), HuffmanNode(105))), HuffmanNode(None, \
    HuffmanNode(103), HuffmanNode(101)))
    >>> avg_length(tree2, freq2)
    True
    >>> improve_tree(tree2, freq2)
    >>> tree2
    True
    >>> avg_length(tree2, freq2)
    True
    """
    items = list(freq_dict.items())
    items.sort(key=lambda x: x[1], reverse=True)
    data = items

    compare_list = []
    for i in data:
        compare_list.append(i[0])

    traverse(tree, compare_list)


def traverse(rootnode, result):
    """
    level order
    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> traverse(tree, result)
    True
    """
    thislevel = [rootnode]
    while thislevel:
        nextlevel = []
        for n in thislevel:
            if n.symbol is not None:
                if len(result) > 0:
                    n.symbol = result[0]
                    result.remove(result[0])
            if n.left:
                nextlevel.append(n.left)
            if n.right:
                nextlevel.append(n.right)
        thislevel = nextlevel


if __name__ == "__main__":
    import python_ta
    python_ta.check_all(config="huffman_pyta.txt")
    # TODO: Uncomment these when you have implemented all the functions
    import doctest
    doctest.testmod()

    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
