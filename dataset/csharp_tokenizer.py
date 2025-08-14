"""
C# Code Tokenizer for HRM Training
Converts C# code into token sequences compatible with HRM's puzzle format
"""

import re
from typing import List, Dict, Set, Tuple, Optional
import json
import numpy as np

class CSharpTokenizer:
    def __init__(self):
        # C# keywords and operators
        self.keywords = {
            'abstract', 'as', 'base', 'bool', 'break', 'byte', 'case', 'catch', 'char', 
            'checked', 'class', 'const', 'continue', 'decimal', 'default', 'delegate', 
            'do', 'double', 'else', 'enum', 'event', 'explicit', 'extern', 'false', 
            'finally', 'fixed', 'float', 'for', 'foreach', 'goto', 'if', 'implicit', 
            'in', 'int', 'interface', 'internal', 'is', 'lock', 'long', 'namespace', 
            'new', 'null', 'object', 'operator', 'out', 'override', 'params', 'private', 
            'protected', 'public', 'readonly', 'ref', 'return', 'sbyte', 'sealed', 
            'short', 'sizeof', 'stackalloc', 'static', 'string', 'struct', 'switch', 
            'this', 'throw', 'true', 'try', 'typeof', 'uint', 'ulong', 'unchecked', 
            'unsafe', 'ushort', 'using', 'virtual', 'void', 'volatile', 'while', 'var'
        }
        
        self.operators = {
            '+', '-', '*', '/', '%', '=', '==', '!=', '<', '>', '<=', '>=', 
            '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '++', '--', 
            '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=',
            '?', ':', '=>', '??', '??='
        }
        
        self.delimiters = {
            '(', ')', '[', ']', '{', '}', ';', ',', '.', '::'
        }
        
        # Build vocabulary
        self.vocab_to_id = {}
        self.id_to_vocab = {}
        self._build_vocab()
        
        # Token patterns for regex tokenization
        self.token_patterns = [
            (r'//.*?\n', 'COMMENT'),
            (r'/\*.*?\*/', 'COMMENT'),
            (r'"(?:[^"\\]|\\.)*"', 'STRING'),
            (r"'(?:[^'\\]|\\.)*'", 'CHAR'),
            (r'\d+\.\d+[fFdDmM]?', 'FLOAT'),
            (r'\d+[lLuUfFdDmM]*', 'NUMBER'),
            (r'[a-zA-Z_][a-zA-Z0-9_]*', 'IDENTIFIER'),
            (r'==|!=|<=|>=|<<|>>|\+\+|--|&&|\|\||=>|\?\?=?', 'OPERATOR'),
            (r'[+\-*/%=<>&|^~!?:]', 'OPERATOR'),
            (r'[(){}\[\];,.]', 'DELIMITER'),
            (r'\s+', 'WHITESPACE'),
        ]
        
        self.compiled_patterns = [(re.compile(pattern, re.DOTALL), token_type) 
                                 for pattern, token_type in self.token_patterns]
    
    def _build_vocab(self):
        """Build vocabulary mapping"""
        vocab_list = ['<PAD>', '<UNK>', '<START>', '<END>', '<MASK>']
        
        # Add special programming tokens
        vocab_list.extend(['<INDENT>', '<DEDENT>', '<NEWLINE>'])
        
        # Add keywords
        vocab_list.extend(sorted(self.keywords))
        
        # Add operators
        vocab_list.extend(sorted(self.operators))
        
        # Add delimiters
        vocab_list.extend(sorted(self.delimiters))
        
        # Add common C# types and methods
        common_tokens = [
            'Console', 'WriteLine', 'ReadLine', 'ToString', 'Parse', 'TryParse',
            'Length', 'Count', 'Add', 'Remove', 'Contains', 'First', 'Last',
            'Where', 'Select', 'OrderBy', 'GroupBy', 'Sum', 'Max', 'Min',
            'List', 'Array', 'Dictionary', 'HashSet', 'Queue', 'Stack',
            'Exception', 'ArgumentException', 'NullReferenceException',
            'Main', 'args', 'value', 'item', 'result', 'temp', 'i', 'j', 'k'
        ]
        vocab_list.extend(common_tokens)
        
        # Reserve space for identifiers, strings, numbers
        for i in range(1000):
            vocab_list.append(f'<ID_{i}>')
        for i in range(500):
            vocab_list.append(f'<STR_{i}>')
        for i in range(500):
            vocab_list.append(f'<NUM_{i}>')
            
        # Build mappings
        for i, token in enumerate(vocab_list):
            self.vocab_to_id[token] = i
            self.id_to_vocab[i] = token
            
        self.vocab_size = len(vocab_list)
        
    def tokenize(self, code: str) -> List[str]:
        """Tokenize C# code into tokens"""
        tokens = []
        pos = 0
        
        while pos < len(code):
            matched = False
            
            for pattern, token_type in self.compiled_patterns:
                match = pattern.match(code, pos)
                if match:
                    token_text = match.group(0)
                    
                    if token_type == 'WHITESPACE':
                        # Handle indentation
                        if '\n' in token_text:
                            tokens.append('<NEWLINE>')
                        # Skip other whitespace
                    elif token_type == 'COMMENT':
                        # Skip comments for now
                        pass
                    elif token_type == 'IDENTIFIER':
                        if token_text in self.keywords:
                            tokens.append(token_text)
                        else:
                            tokens.append(f'<ID_{hash(token_text) % 1000}>')
                    elif token_type == 'STRING':
                        tokens.append(f'<STR_{hash(token_text) % 500}>')
                    elif token_type in ['NUMBER', 'FLOAT']:
                        tokens.append(f'<NUM_{hash(token_text) % 500}>')
                    else:
                        tokens.append(token_text)
                    
                    pos = match.end()
                    matched = True
                    break
            
            if not matched:
                # Skip unknown character
                pos += 1
                
        return tokens
    
    def encode(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs"""
        return [self.vocab_to_id.get(token, self.vocab_to_id['<UNK>']) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> List[str]:
        """Convert IDs back to tokens"""
        return [self.id_to_vocab.get(tid, '<UNK>') for tid in token_ids]
    
    def encode_code_pair(self, input_code: str, target_code: str, max_length: int = 1024) -> Tuple[List[int], List[int]]:
        """Encode input/target code pair for training"""
        input_tokens = ['<START>'] + self.tokenize(input_code) + ['<END>']
        target_tokens = ['<START>'] + self.tokenize(target_code) + ['<END>']
        
        # Truncate if too long
        if len(input_tokens) > max_length:
            input_tokens = input_tokens[:max_length-1] + ['<END>']
        if len(target_tokens) > max_length:
            target_tokens = target_tokens[:max_length-1] + ['<END>']
            
        # Pad to max_length
        input_ids = self.encode(input_tokens)
        target_ids = self.encode(target_tokens)
        
        # Pad
        while len(input_ids) < max_length:
            input_ids.append(self.vocab_to_id['<PAD>'])
        while len(target_ids) < max_length:
            target_ids.append(self.vocab_to_id['<PAD>'])
            
        return input_ids, target_ids

# Test the tokenizer
if __name__ == "__main__":
    tokenizer = CSharpTokenizer()
    
    sample_code = '''
    public class HelloWorld 
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");
        }
    }
    '''
    
    tokens = tokenizer.tokenize(sample_code)
    print("Tokens:", tokens)
    print("Vocab size:", tokenizer.vocab_size)