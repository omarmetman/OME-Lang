#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re
import math
import random
import datetime
import json
import os
import time
import threading
from typing import Dict, Any, List, Union, Optional, Callable
from enum import Enum
from dataclasses import dataclass

class TokenType(Enum):
    # Keywords
    LET = 'let'
    PRINT = 'print'
    IF = 'if'
    ELSE = 'else'
    FOR = 'for'
    WHILE = 'while'
    FUNCTION = 'function'
    RETURN = 'return'
    CLASS = 'class'
    IMPORT = 'import'
    TRY = 'try'
    CATCH = 'catch'
    THROW = 'throw'
    NULL = 'null'
    TRUE = 'true'
    FALSE = 'false'
    
    # Operators
    PLUS = '+'
    MINUS = '-'
    MULTIPLY = '*'
    DIVIDE = '/'
    MODULO = '%'
    POWER = '^'
    EQUALS = '=='
    NOT_EQUALS = '!='
    LESS = '<'
    GREATER = '>'
    LESS_EQUAL = '<='
    GREATER_EQUAL = '>='
    AND = '&&'
    OR = '||'
    NOT = '!'
    ASSIGN = '='
    
    # Delimiters
    LPAREN = '('
    RPAREN = ')'
    LBRACE = '{'
    RBRACE = '}'
    LBRACKET = '['
    RBRACKET = ']'
    COMMA = ','
    DOT = '.'
    COLON = ':'
    SEMICOLON = ';'
    
    # Literals
    IDENTIFIER = 'IDENTIFIER'
    NUMBER = 'NUMBER'
    STRING = 'STRING'
    EOF = 'EOF'

@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    column: int

class ASTNode:
    pass

@dataclass
class Program(ASTNode):
    statements: List[ASTNode]

@dataclass
class VariableDeclaration(ASTNode):
    name: str
    value: ASTNode

@dataclass
class FunctionDeclaration(ASTNode):
    name: str
    params: List[str]
    body: List[ASTNode]

@dataclass
class ClassDeclaration(ASTNode):
    name: str
    methods: List[FunctionDeclaration]
    properties: List[VariableDeclaration]

@dataclass
class BinaryExpression(ASTNode):
    left: ASTNode
    operator: str
    right: ASTNode

@dataclass
class UnaryExpression(ASTNode):
    operator: str
    argument: ASTNode

@dataclass
class Literal(ASTNode):
    value: Any

@dataclass
class Identifier(ASTNode):
    name: str

@dataclass
class CallExpression(ASTNode):
    callee: ASTNode
    arguments: List[ASTNode]

@dataclass
class MemberExpression(ASTNode):
    object: ASTNode
    property: ASTNode
    computed: bool

@dataclass
class IfStatement(ASTNode):
    condition: ASTNode
    consequent: List[ASTNode]
    alternate: Optional[List[ASTNode]]

@dataclass
class ForStatement(ASTNode):
    init: Optional[ASTNode]
    condition: Optional[ASTNode]
    update: Optional[ASTNode]
    body: List[ASTNode]

@dataclass
class WhileStatement(ASTNode):
    condition: ASTNode
    body: List[ASTNode]

@dataclass
class ReturnStatement(ASTNode):
    argument: Optional[ASTNode]

@dataclass
class TryCatchStatement(ASTNode):
    try_block: List[ASTNode]
    catch_block: List[ASTNode]
    error_var: Optional[str]

class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
        self.keywords = {
            'let': TokenType.LET,
            'print': TokenType.PRINT,
            'if': TokenType.IF,
            'else': TokenType.ELSE,
            'for': TokenType.FOR,
            'while': TokenType.WHILE,
            'function': TokenType.FUNCTION,
            'return': TokenType.RETURN,
            'class': TokenType.CLASS,
            'import': TokenType.IMPORT,
            'try': TokenType.TRY,
            'catch': TokenType.CATCH,
            'throw': TokenType.THROW,
            'null': TokenType.NULL,
            'true': TokenType.TRUE,
            'false': TokenType.FALSE
        }

    def next_token(self) -> Token:
        while self.position < len(self.source) and self.source[self.position].isspace():
            if self.source[self.position] == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            self.position += 1

        if self.position >= len(self.source):
            return Token(TokenType.EOF, None, self.line, self.column)

        char = self.source[self.position]

        # Handle comments
        if char == '#':
            while self.position < len(self.source) and self.source[self.position] != '\n':
                self.position += 1
                self.column += 1
            return self.next_token()

        # Handle strings
        if char in ['"', "'"]:
            return self.read_string(char)

        # Handle numbers
        if char.isdigit():
            return self.read_number()

        # Handle identifiers and keywords
        if char.isalpha() or char == '_':
            return self.read_identifier()

        # Handle operators
        operators = {
            '+': TokenType.PLUS,
            '-': TokenType.MINUS,
            '*': TokenType.MULTIPLY,
            '/': TokenType.DIVIDE,
            '%': TokenType.MODULO,
            '^': TokenType.POWER,
            '=': TokenType.ASSIGN,
            '!': TokenType.NOT,
            '<': TokenType.LESS,
            '>': TokenType.GREATER,
            '&': TokenType.AND,
            '|': TokenType.OR,
            '(': TokenType.LPAREN,
            ')': TokenType.RPAREN,
            '{': TokenType.LBRACE,
            '}': TokenType.RBRACE,
            '[': TokenType.LBRACKET,
            ']': TokenType.RBRACKET,
            ',': TokenType.COMMA,
            '.': TokenType.DOT,
            ':': TokenType.COLON,
            ';': TokenType.SEMICOLON
        }

        if char in operators:
            # Handle multi-character operators
            if char == '=' and self.peek() == '=':
                self.position += 1
                self.column += 1
                token = Token(TokenType.EQUALS, '==', self.line, self.column)
            elif char == '!' and self.peek() == '=':
                self.position += 1
                self.column += 1
                token = Token(TokenType.NOT_EQUALS, '!=', self.line, self.column)
            elif char == '<' and self.peek() == '=':
                self.position += 1
                self.column += 1
                token = Token(TokenType.LESS_EQUAL, '<=', self.line, self.column)
            elif char == '>' and self.peek() == '=':
                self.position += 1
                self.column += 1
                token = Token(TokenType.GREATER_EQUAL, '>=', self.line, self.column)
            elif char == '&' and self.peek() == '&':
                self.position += 1
                self.column += 1
                token = Token(TokenType.AND, '&&', self.line, self.column)
            elif char == '|' and self.peek() == '|':
                self.position += 1
                self.column += 1
                token = Token(TokenType.OR, '||', self.line, self.column)
            else:
                token = Token(operators[char], char, self.line, self.column)
            
            self.position += 1
            self.column += 1
            return token

        # Unknown character
        self.position += 1
        self.column += 1
        return Token(TokenType.IDENTIFIER, char, self.line, self.column)

    def read_string(self, quote_char: str) -> Token:
        start_line = self.line
        start_column = self.column
        self.position += 1
        self.column += 1
        value = ''
        
        while self.position < len(self.source) and self.source[self.position] != quote_char:
            if self.source[self.position] == '\\':
                self.position += 1
                self.column += 1
                if self.position < len(self.source):
                    escape_char = self.source[self.position]
                    if escape_char == 'n':
                        value += '\n'
                    elif escape_char == 't':
                        value += '\t'
                    elif escape_char == 'r':
                        value += '\r'
                    else:
                        value += escape_char
            else:
                value += self.source[self.position]
            
            if self.source[self.position] == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            self.position += 1

        if self.position < len(self.source):
            self.position += 1
            self.column += 1

        return Token(TokenType.STRING, value, start_line, start_column)

    def read_number(self) -> Token:
        start_line = self.line
        start_column = self.column
        value = ''
        has_dot = False

        while self.position < len(self.source) and (self.source[self.position].isdigit() or 
                                                  (self.source[self.position] == '.' and not has_dot)):
            if self.source[self.position] == '.':
                has_dot = True
            value += self.source[self.position]
            self.position += 1
            self.column += 1

        number = float(value) if has_dot else int(value)
        return Token(TokenType.NUMBER, number, start_line, start_column)

    def read_identifier(self) -> Token:
        start_line = self.line
        start_column = self.column
        value = ''

        while self.position < len(self.source) and (self.source[self.position].isalnum() or 
                                                  self.source[self.position] == '_'):
            value += self.source[self.position]
            self.position += 1
            self.column += 1

        token_type = self.keywords.get(value, TokenType.IDENTIFIER)
        return Token(token_type, value, start_line, start_column)

    def peek(self) -> str:
        if self.position + 1 < len(self.source):
            return self.source[self.position + 1]
        return '\0'

class Parser:
    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.current_token = self.lexer.next_token()
        self.previous_token = None

    def eat(self, token_type: TokenType):
        if self.current_token.type == token_type:
            self.previous_token = self.current_token
            self.current_token = self.lexer.next_token()
        else:
            raise SyntaxError(f"Expected {token_type}, got {self.current_token.type}")

    def parse(self) -> Program:
        statements = []
        while self.current_token.type != TokenType.EOF:
            statements.append(self.parse_statement())
        return Program(statements)

    def parse_statement(self) -> ASTNode:
        token = self.current_token
        
        if token.type == TokenType.LET:
            return self.parse_variable_declaration()
        elif token.type == TokenType.PRINT:
            return self.parse_print_statement()
        elif token.type == TokenType.IF:
            return self.parse_if_statement()
        elif token.type == TokenType.FOR:
            return self.parse_for_statement()
        elif token.type == TokenType.WHILE:
            return self.parse_while_statement()
        elif token.type == TokenType.FUNCTION:
            return self.parse_function_declaration()
        elif token.type == TokenType.CLASS:
            return self.parse_class_declaration()
        elif token.type == TokenType.RETURN:
            return self.parse_return_statement()
        elif token.type == TokenType.TRY:
            return self.parse_try_catch_statement()
        elif token.type == TokenType.IMPORT:
            return self.parse_import_statement()
        else:
            return self.parse_expression()

    def parse_variable_declaration(self) -> VariableDeclaration:
        self.eat(TokenType.LET)
        name = self.current_token.value
        self.eat(TokenType.IDENTIFIER)
        self.eat(TokenType.ASSIGN)
        value = self.parse_expression()
        return VariableDeclaration(name, value)

    def parse_print_statement(self) -> CallExpression:
        self.eat(TokenType.PRINT)
        args = [self.parse_expression()]
        return CallExpression(Identifier('print'), args)

    def parse_if_statement(self) -> IfStatement:
        self.eat(TokenType.IF)
        condition = self.parse_expression()
        consequent = self.parse_block()
        alternate = None
        
        if self.current_token.type == TokenType.ELSE:
            self.eat(TokenType.ELSE)
            alternate = self.parse_block()
            
        return IfStatement(condition, consequent, alternate)

    def parse_for_statement(self) -> ForStatement:
        self.eat(TokenType.FOR)
        
        # Parse initialization
        init = None
        if self.current_token.type != TokenType.SEMICOLON:
            init = self.parse_statement()
        self.eat(TokenType.SEMICOLON)
        
        # Parse condition
        condition = None
        if self.current_token.type != TokenType.SEMICOLON:
            condition = self.parse_expression()
        self.eat(TokenType.SEMICOLON)
        
        # Parse update
        update = None
        if self.current_token.type != TokenType.LBRACE:
            update = self.parse_expression()
        
        body = self.parse_block()
        return ForStatement(init, condition, update, body)

    def parse_while_statement(self) -> WhileStatement:
        self.eat(TokenType.WHILE)
        condition = self.parse_expression()
        body = self.parse_block()
        return WhileStatement(condition, body)

    def parse_function_declaration(self) -> FunctionDeclaration:
        self.eat(TokenType.FUNCTION)
        name = self.current_token.value
        self.eat(TokenType.IDENTIFIER)
        self.eat(TokenType.LPAREN)
        
        params = []
        if self.current_token.type != TokenType.RPAREN:
            params.append(self.current_token.value)
            self.eat(TokenType.IDENTIFIER)
            while self.current_token.type == TokenType.COMMA:
                self.eat(TokenType.COMMA)
                params.append(self.current_token.value)
                self.eat(TokenType.IDENTIFIER)
        
        self.eat(TokenType.RPAREN)
        body = self.parse_block()
        return FunctionDeclaration(name, params, body)

    def parse_class_declaration(self) -> ClassDeclaration:
        self.eat(TokenType.CLASS)
        name = self.current_token.value
        self.eat(TokenType.IDENTIFIER)
        self.eat(TokenType.LBRACE)
        
        methods = []
        properties = []
        
        while self.current_token.type != TokenType.RBRACE:
            if self.current_token.type == TokenType.FUNCTION:
                methods.append(self.parse_function_declaration())
            elif self.current_token.type == TokenType.LET:
                properties.append(self.parse_variable_declaration())
            else:
                self.eat(TokenType.IDENTIFIER)  # Skip unexpected tokens
        
        self.eat(TokenType.RBRACE)
        return ClassDeclaration(name, methods, properties)

    def parse_return_statement(self) -> ReturnStatement:
        self.eat(TokenType.RETURN)
        argument = None
        if self.current_token.type != TokenType.SEMICOLON:
            argument = self.parse_expression()
        return ReturnStatement(argument)

    def parse_try_catch_statement(self) -> TryCatchStatement:
        self.eat(TokenType.TRY)
        try_block = self.parse_block()
        self.eat(TokenType.CATCH)
        self.eat(TokenType.LPAREN)
        error_var = self.current_token.value
        self.eat(TokenType.IDENTIFIER)
        self.eat(TokenType.RPAREN)
        catch_block = self.parse_block()
        return TryCatchStatement(try_block, catch_block, error_var)

    def parse_import_statement(self) -> ASTNode:
        self.eat(TokenType.IMPORT)
        module_path = self.current_token.value
        self.eat(TokenType.STRING)
        return CallExpression(Identifier('import'), [Literal(module_path)])

    def parse_block(self) -> List[ASTNode]:
        self.eat(TokenType.LBRACE)
        statements = []
        
        while self.current_token.type != TokenType.RBRACE:
            statements.append(self.parse_statement())
            if self.current_token.type == TokenType.SEMICOLON:
                self.eat(TokenType.SEMICOLON)
        
        self.eat(TokenType.RBRACE)
        return statements

    def parse_expression(self) -> ASTNode:
        return self.parse_assignment()

    def parse_assignment(self) -> ASTNode:
        left = self.parse_logical_or()
        
        if self.current_token.type == TokenType.ASSIGN:
            self.eat(TokenType.ASSIGN)
            right = self.parse_assignment()
            return BinaryExpression(left, '=', right)
        
        return left

    def parse_logical_or(self) -> ASTNode:
        left = self.parse_logical_and()
        
        while self.current_token.type == TokenType.OR:
            operator = self.current_token.value
            self.eat(TokenType.OR)
            right = self.parse_logical_and()
            left = BinaryExpression(left, operator, right)
        
        return left

    def parse_logical_and(self) -> ASTNode:
        left = self.parse_equality()
        
        while self.current_token.type == TokenType.AND:
            operator = self.current_token.value
            self.eat(TokenType.AND)
            right = self.parse_equality()
            left = BinaryExpression(left, operator, right)
        
        return left

    def parse_equality(self) -> ASTNode:
        left = self.parse_comparison()
        
        while self.current_token.type in [TokenType.EQUALS, TokenType.NOT_EQUALS]:
            operator = self.current_token.value
            self.eat(self.current_token.type)
            right = self.parse_comparison()
            left = BinaryExpression(left, operator, right)
        
        return left

    def parse_comparison(self) -> ASTNode:
        left = self.parse_term()
        
        while self.current_token.type in [TokenType.LESS, TokenType.GREATER, 
                                         TokenType.LESS_EQUAL, TokenType.GREATER_EQUAL]:
            operator = self.current_token.value
            self.eat(self.current_token.type)
            right = self.parse_term()
            left = BinaryExpression(left, operator, right)
        
        return left

    def parse_term(self) -> ASTNode:
        left = self.parse_factor()
        
        while self.current_token.type in [TokenType.PLUS, TokenType.MINUS]:
            operator = self.current_token.value
            self.eat(self.current_token.type)
            right = self.parse_factor()
            left = BinaryExpression(left, operator, right)
        
        return left

    def parse_factor(self) -> ASTNode:
        left = self.parse_power()
        
        while self.current_token.type in [TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MODULO]:
            operator = self.current_token.value
            self.eat(self.current_token.type)
            right = self.parse_power()
            left = BinaryExpression(left, operator, right)
        
        return left

    def parse_power(self) -> ASTNode:
        left = self.parse_unary()
        
        while self.current_token.type == TokenType.POWER:
            operator = self.current_token.value
            self.eat(TokenType.POWER)
            right = self.parse_unary()
            left = BinaryExpression(left, operator, right)
        
        return left

    def parse_unary(self) -> ASTNode:
        if self.current_token.type in [TokenType.MINUS, TokenType.NOT]:
            operator = self.current_token.value
            self.eat(self.current_token.type)
            argument = self.parse_unary()
            return UnaryExpression(operator, argument)
        
        return self.parse_primary()

    def parse_primary(self) -> ASTNode:
        token = self.current_token
        
        if token.type == TokenType.NUMBER:
            self.eat(TokenType.NUMBER)
            return Literal(token.value)
        elif token.type == TokenType.STRING:
            self.eat(TokenType.STRING)
            return Literal(token.value)
        elif token.type == TokenType.TRUE:
            self.eat(TokenType.TRUE)
            return Literal(True)
        elif token.type == TokenType.FALSE:
            self.eat(TokenType.FALSE)
            return Literal(False)
        elif token.type == TokenType.NULL:
            self.eat(TokenType.NULL)
            return Literal(None)
        elif token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            expression = self.parse_expression()
            self.eat(TokenType.RPAREN)
            return expression
        elif token.type == TokenType.IDENTIFIER:
            name = token.value
            self.eat(TokenType.IDENTIFIER)
            
            if self.current_token.type == TokenType.LPAREN:
                self.eat(TokenType.LPAREN)
                arguments = []
                
                if self.current_token.type != TokenType.RPAREN:
                    arguments.append(self.parse_expression())
                    while self.current_token.type == TokenType.COMMA:
                        self.eat(TokenType.COMMA)
                        arguments.append(self.parse_expression())
                
                self.eat(TokenType.RPAREN)
                return CallExpression(Identifier(name), arguments)
            elif self.current_token.type in [TokenType.DOT, TokenType.LBRACKET]:
                return self.parse_member_expression(Identifier(name))
            else:
                return Identifier(name)
        
        raise SyntaxError(f"Unexpected token: {token.type}")

    def parse_member_expression(self, object: ASTNode) -> ASTNode:
        while self.current_token.type in [TokenType.DOT, TokenType.LBRACKET]:
            if self.current_token.type == TokenType.DOT:
                self.eat(TokenType.DOT)
                property = Identifier(self.current_token.value)
                self.eat(TokenType.IDENTIFIER)
                object = MemberExpression(object, property, False)
            else:
                self.eat(TokenType.LBRACKET)
                property = self.parse_expression()
                self.eat(TokenType.RBRACKET)
                object = MemberExpression(object, property, True)
        
        return object

class OMEInterpreter:
    def __init__(self):
        self.variables = {
            'PI': math.pi,
            'E': math.e,
            'TRUE': True,
            'FALSE': False,
            'NULL': None,
            'VERSION': '2.0.0'
        }
        self.functions = self._create_builtin_functions()
        self.classes = {}
        self.scope = [self.variables.copy()]
        self.output = []
        self.debug_mode = False
        
    def _create_builtin_functions(self) -> Dict[str, Callable]:
        return {
            'print': self._print,
            'input': self._input,
            'length': self._length,
            'upper': self._upper,
            'lower': self._lower,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'sqrt': math.sqrt,
            'abs': abs,
            'round': round,
            'random': random.random,
            'now': datetime.datetime.now,
            'time': time.time,
            'sleep': time.sleep,
            'json_stringify': json.dumps,
            'json_parse': json.loads,
            'type': self._type,
            'range': range,
            'map': self._map,
            'filter': self._filter,
            'reduce': self._reduce,
            'import': self._import,
            'exit': sys.exit,
            'debug': self._debug
        }
    
    def _print(self, *args):
        result = ' '.join(str(arg) for arg in args)
        self.output.append(result)
        print(result)
        return result
    
    def _input(self, prompt=""):
        return input(prompt)
    
    def _length(self, value):
        return len(str(value))
    
    def _upper(self, value):
        return str(value).upper()
    
    def _lower(self, value):
        return str(value).lower()
    
    def _type(self, value):
        return type(value).__name__
    
    def _map(self, func, iterable):
        return [self._call_function(func, [item]) for item in iterable]
    
    def _filter(self, func, iterable):
        return [item for item in iterable if self._call_function(func, [item])]
    
    def _reduce(self, func, iterable, initial=None):
        result = initial if initial is not None else iterable[0]
        start = 1 if initial is not None else 1
        for item in iterable[start:]:
            result = self._call_function(func, [result, item])
        return result
    
    def _import(self, module_path):
        if not module_path.endswith('.ome'):
            module_path += '.ome'
        
        try:
            with open(module_path, 'r', encoding='utf-8') as file:
                code = file.read()
            
            lexer = Lexer(code)
            parser = Parser(lexer)
            program = parser.parse()
            
            # Create new scope for module
            self.enter_scope()
            self.interpret(program)
            module_vars = self.scope[-1].copy()
            self.exit_scope()
            
            return module_vars
        except FileNotFoundError:
            raise ImportError(f"Cannot import '{module_path}' - File not found")
    
    def _debug(self, *args):
        if self.debug_mode:
            self._print("[DEBUG]", *args)
    
    def enter_scope(self):
        self.scope.append({})
    
    def exit_scope(self):
        if len(self.scope) > 1:
            self.scope.pop()
    
    def set_variable(self, name: str, value: Any):
        # Check current scope first, then global scope
        for scope in reversed(self.scope):
            if name in scope:
                scope[name] = value
                return
        self.scope[-1][name] = value
    
    def get_variable(self, name: str) -> Any:
        for scope in reversed(self.scope):
            if name in scope:
                return scope[name]
        raise NameError(f"Variable '{name}' not found")
    
    def _call_function(self, func: Callable, args: List[Any]) -> Any:
        if callable(func):
            return func(*args)
        raise TypeError(f"{func} is not callable")
    
    def interpret(self, node: ASTNode) -> Any:
        method_name = f'interpret_{type(node).__name__}'
        method = getattr(self, method_name, self._interpret_unknown)
        return method(node)
    
    def _interpret_unknown(self, node: ASTNode) -> Any:
        raise TypeError(f"Unknown node type: {type(node)}")
    
    def interpret_Program(self, node: Program) -> Any:
        result = None
        for statement in node.statements:
            result = self.interpret(statement)
        return result
    
    def interpret_VariableDeclaration(self, node: VariableDeclaration) -> Any:
        value = self.interpret(node.value)
        self.set_variable(node.name, value)
        return value
    
    def interpret_FunctionDeclaration(self, node: FunctionDeclaration) -> Any:
        def ome_function(*args):
            self.enter_scope()
            for param, arg in zip(node.params, args):
                self.set_variable(param, arg)
            
            result = None
            for stmt in node.body:
                result = self.interpret(stmt)
                if isinstance(result, ReturnStatement):
                    result = self.interpret(result.argument) if result.argument else None
                    break
            
            self.exit_scope()
            return result
        
        self.set_variable(node.name, ome_function)
        return ome_function
    
    def interpret_ClassDeclaration(self, node: ClassDeclaration) -> Any:
        class_dict = {}
        
        # Add properties
        for prop in node.properties:
            value = self.interpret(prop.value)
            class_dict[prop.name] = value
        
        # Add methods
        for method in node.methods:
            def create_method(method_node):
                def ome_method(self_ref, *args):
                    self.enter_scope()
                    self.set_variable('this', self_ref)
                    
                    for param, arg in zip(method_node.params, args):
                        self.set_variable(param, arg)
                    
                    result = None
                    for stmt in method_node.body:
                        result = self.interpret(stmt)
                        if isinstance(result, ReturnStatement):
                            result = self.interpret(result.argument) if result.argument else None
                            break
                    
                    self.exit_scope()
                    return result
                return ome_method
            
            class_dict[method.name] = create_method(method)
        
        self.classes[node.name] = class_dict
        self.set_variable(node.name, class_dict)
        return class_dict
    
    def interpret_BinaryExpression(self, node: BinaryExpression) -> Any:
        left = self.interpret(node.left)
        right = self.interpret(node.right)
        
        if node.operator == '+':
            if isinstance(left, str) or isinstance(right, str):
                return str(left) + str(right)
            return left + right
        elif node.operator == '-':
            return left - right
        elif node.operator == '*':
            return left * right
        elif node.operator == '/':
            return left / right
        elif node.operator == '%':
            return left % right
        elif node.operator == '^':
            return left ** right
        elif node.operator == '==':
            return left == right
        elif node.operator == '!=':
            return left != right
        elif node.operator == '<':
            return left < right
        elif node.operator == '>':
            return left > right
        elif node.operator == '<=':
            return left <= right
        elif node.operator == '>=':
            return left >= right
        elif node.operator == '&&':
            return left and right
        elif node.operator == '||':
            return left or right
        elif node.operator == '=':
            if isinstance(node.left, Identifier):
                self.set_variable(node.left.name, right)
                return right
            raise SyntaxError("Invalid assignment")
        
        raise SyntaxError(f"Unknown operator: {node.operator}")
    
    def interpret_UnaryExpression(self, node: UnaryExpression) -> Any:
        argument = self.interpret(node.argument)
        
        if node.operator == '-':
            return -argument
        elif node.operator == '!':
            return not argument
        
        raise SyntaxError(f"Unknown unary operator: {node.operator}")
    
    def interpret_Literal(self, node: Literal) -> Any:
        return node.value
    
    def interpret_Identifier(self, node: Identifier) -> Any:
        if node.name in self.functions:
            return self.functions[node.name]
        return self.get_variable(node.name)
    
    def interpret_CallExpression(self, node: CallExpression) -> Any:
        callee = self.interpret(node.callee)
        args = [self.interpret(arg) for arg in node.arguments]
        
        if callable(callee):
            return callee(*args)
        raise TypeError(f"{callee} is not callable")
    
    def interpret_MemberExpression(self, node: MemberExpression) -> Any:
        obj = self.interpret(node.object)
        prop = self.interpret(node.property)
        
        if isinstance(obj, dict) and prop in obj:
            return obj[prop]
        elif hasattr(obj, prop):
            return getattr(obj, prop)
        
        raise AttributeError(f"Object has no attribute '{prop}'")
    
    def interpret_IfStatement(self, node: IfStatement) -> Any:
        condition = self.interpret(node.condition)
        
        if condition:
            self.enter_scope()
            for stmt in node.consequent:
                result = self.interpret(stmt)
            self.exit_scope()
            return result
        elif node.alternate:
            self.enter_scope()
            for stmt in node.alternate:
                result = self.interpret(stmt)
            self.exit_scope()
            return result
        
        return None
    
    def interpret_ForStatement(self, node: ForStatement) -> Any:
        result = None
        self.enter_scope()
        
        if node.init:
            self.interpret(node.init)
        
        while True:
            if node.condition:
                condition = self.interpret(node.condition)
                if not condition:
                    break
            
            for stmt in node.body:
                result = self.interpret(stmt)
                if isinstance(result, ReturnStatement):
                    break
            
            if isinstance(result, ReturnStatement):
                break
            
            if node.update:
                self.interpret(node.update)
        
        self.exit_scope()
        return result
    
    def interpret_WhileStatement(self, node: WhileStatement) -> Any:
        result = None
        
        while self.interpret(node.condition):
            self.enter_scope()
            for stmt in node.body:
                result = self.interpret(stmt)
                if isinstance(result, ReturnStatement):
                    break
            self.exit_scope()
            
            if isinstance(result, ReturnStatement):
                break
        
        return result
    
    def interpret_ReturnStatement(self, node: ReturnStatement) -> Any:
        if node.argument:
            return ReturnStatement(self.interpret(node.argument))
        return ReturnStatement(None)
    
    def interpret_TryCatchStatement(self, node: TryCatchStatement) -> Any:
        try:
            self.enter_scope()
            for stmt in node.try_block:
                result = self.interpret(stmt)
            self.exit_scope()
            return result
        except Exception as e:
            self.enter_scope()
            if node.error_var:
                self.set_variable(node.error_var, str(e))
            for stmt in node.catch_block:
                result = self.interpret(stmt)
            self.exit_scope()
            return result

def main():
    if len(sys.argv) < 2:
        print("""
   ██████╗  ███╗   ███╗ ███████╗
  ██╔═══██╗ ████╗ ████║ ██╔════╝
  ██║   ██║ ██╔████╔██║ █████╗  
  ██║   ██║ ██║╚██╔╝██║ ██╔══╝  
  ╚██████╔╝ ██║ ╚═╝ ██║ ███████╗
   ╚═════╝  ╚═╝     ╚═╝ ╚══════╝
             O   M   E  v2.0
                             
Usage: python ome.py program.ome [--debug]
        """)
        sys.exit(1)
    
    filename = sys.argv[1]
    debug_mode = '--debug' in sys.argv
    
    if not filename.endswith('.ome'):
        print("Error: File must have .ome extension")
        sys.exit(1)
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            code = file.read()
        
        # Lexing
        lexer = Lexer(code)
        
        # Parsing
        parser = Parser(lexer)
        program = parser.parse()
        
        # Interpretation
        interpreter = OMEInterpreter()
        interpreter.debug_mode = debug_mode
        interpreter.interpret(program)
        
        if debug_mode:
            print("\n=== DEBUG INFO ===")
            print("Variables:", interpreter.variables)
            print("Output:", interpreter.output)
            print("Classes:", list(interpreter.classes.keys()))
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if debug_mode:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
