
import numpy as np
from capstone import *

md = Cs(CS_ARCH_X86, CS_MODE_32)
class Ins(object):
    def __init__(self, address, binary, string):
        self.address = address
        self.binary = binary
        self.string = string
    
    def get_disas(self):
        length  = len(self.binary.replace(b' ', b''))
        aint =int(self.binary.replace(b' ', b''), 16) 
        # print(aint)
        abytes = aint.to_bytes(int(length/2), 'big')
      
        for (address, size, mnemonic, op_str) in md.disasm_lite(abytes, self.address):
            # print(self.binary, self.string, "-->", address, mnemonic, op_str.replace(',', '').replace('[', '[ ').replace(']', ' ]').replace(':', ' : '))
            return mnemonic + " " + op_str.replace(',', '').replace('[', '[ ').replace(']', ' ]').replace(':', ' : ').replace('*', ' * ')

        return "nop"

    def _print(self):
        print("%x\t\t%s\t\t\t%s" % (self.address, self.binary, self.string))

class BasicBlock(object):
    def __init__(self):
        self.ready = False
        self.ids = []
        self.instructions = []

    def set_id(self, id):
        if len(self.ids) == 3:
            self.ids.append(id*2)
        else:
            self.ids.append(id)
        if len(self.ids) == 4:
            self.ready = True

    def get_binary(self):
        uint8 = []
        for ins in self.instructions:
            b = ins.binary
            for i in range(len(b)):
                uint8.append(b[i])
        return uint8

    def set_ids_for_prologue(self, ids):
        self.ids = []
        self.ids.append(ids[0])
        self.ids.append(ids[1])
        self.ids.append(0)
        self.ids.append(0)
        self.ready = True

    def set_ids_for_after_branch(self, ids):
        self.ids = []
        self.ids.append(ids[0])
        self.ids.append(ids[1])
        self.ids.append(ids[2])
        self.ids.append(ids[3]+1)
        self.ready = True

    def clear_id(self):
        self.ids = []

    def can_receive_ins(self):
        return self.ready
    
    def get_signature(self):
        signature = 0
        for id in self.ids:
            signature += id
        return signature

    def _print(self):
        # if len(self.ids) == 4:
        #     print("---Basic Block (IDS:[%x][%x][%x][%x])---" % (self.ids[0], self.ids[1], self.ids[2], self.ids[3]))
        # else:
        #     print("---Basic Block Begin---")
        for ins in self.instructions:
            ins._print()
        # print("---Basic Block End---\n")


class Function(object):
    def __init__(self, name):
        self.name = name
        self.basicblocks = []

    def add_bb(self, bb):
        self.basicblocks.append(bb)

    def _print(self):
        print("---Function: %s ---" % self.name)
        for bb in self.basicblocks:
            bb._print()
    
    def get_first_n_bytes(self, n):
        uint8 = []
        for bb in self.basicblocks:
            uint8.extend(bb.get_binary())
            if n != None and len(uint8) >= n:
                return uint8[:n]
        if n != None:
            uint8.extend([0]*(n-len(uint8)))
        return np.array(uint8)
    
    def get_first_n_ins(self, n):
        insstr = []
        for bb in self.basicblocks:
            for ins in bb.instructions:
                # print(ins.get_disas())
                
                insstr.append(ins.get_disas())
                n -= 1
                if n == 0:
                    return insstr
        return insstr

    def get_last_n_ins(self, n):
        insstr = []
        for bb in reversed(self.basicblocks):
            bbins = []
            for ins in bb.instructions:
                bbins.append(ins.get_disas())
            insstr = bbins + insstr
            if len(insstr) >= n:
                break
        return insstr[len(insstr)-n:]


    def get_ins_num(self):
        count = 0
        for bb in self.basicblocks:
            count += len(bb.instructions)
        return count
