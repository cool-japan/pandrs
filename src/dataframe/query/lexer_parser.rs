//! Lexical analysis and parsing for query expressions
//!
//! This module provides lexical analysis (tokenization) and parsing functionality
//! to convert query strings into abstract syntax trees (AST).

use std::iter::Peekable;
use std::str::Chars;

use super::ast::{BinaryOp, Expr, LiteralValue, Token, UnaryOp};
use crate::core::error::{Error, Result};

/// Lexer for tokenizing query expressions
pub struct Lexer {
    chars: Peekable<Chars<'static>>,
    input: &'static str,
}

impl Lexer {
    /// Create a new lexer
    pub fn new(input: &'static str) -> Self {
        Self {
            chars: input.chars().peekable(),
            input,
        }
    }

    /// Get the next token
    pub fn next_token(&mut self) -> Result<Token> {
        self.skip_whitespace();

        match self.chars.peek() {
            None => Ok(Token::Eof),
            Some(&ch) => match ch {
                '(' => {
                    self.chars.next();
                    Ok(Token::LeftParen)
                }
                ')' => {
                    self.chars.next();
                    Ok(Token::RightParen)
                }
                ',' => {
                    self.chars.next();
                    Ok(Token::Comma)
                }
                '+' => {
                    self.chars.next();
                    Ok(Token::Plus)
                }
                '-' => {
                    self.chars.next();
                    Ok(Token::Minus)
                }
                '*' => {
                    self.chars.next();
                    if self.chars.peek() == Some(&'*') {
                        self.chars.next();
                        Ok(Token::Power)
                    } else {
                        Ok(Token::Multiply)
                    }
                }
                '/' => {
                    self.chars.next();
                    Ok(Token::Divide)
                }
                '%' => {
                    self.chars.next();
                    Ok(Token::Modulo)
                }
                '=' => {
                    self.chars.next();
                    if self.chars.peek() == Some(&'=') {
                        self.chars.next();
                        Ok(Token::Equal)
                    } else {
                        Err(Error::InvalidValue(
                            "Expected '==' for equality comparison".to_string(),
                        ))
                    }
                }
                '!' => {
                    self.chars.next();
                    if self.chars.peek() == Some(&'=') {
                        self.chars.next();
                        Ok(Token::NotEqual)
                    } else {
                        Ok(Token::Not)
                    }
                }
                '<' => {
                    self.chars.next();
                    if self.chars.peek() == Some(&'=') {
                        self.chars.next();
                        Ok(Token::LessThanOrEqual)
                    } else {
                        Ok(Token::LessThan)
                    }
                }
                '>' => {
                    self.chars.next();
                    if self.chars.peek() == Some(&'=') {
                        self.chars.next();
                        Ok(Token::GreaterThanOrEqual)
                    } else {
                        Ok(Token::GreaterThan)
                    }
                }
                '&' => {
                    self.chars.next();
                    if self.chars.peek() == Some(&'&') {
                        self.chars.next();
                        Ok(Token::And)
                    } else {
                        Err(Error::InvalidValue(
                            "Expected '&&' for logical AND".to_string(),
                        ))
                    }
                }
                '|' => {
                    self.chars.next();
                    if self.chars.peek() == Some(&'|') {
                        self.chars.next();
                        Ok(Token::Or)
                    } else {
                        Err(Error::InvalidValue(
                            "Expected '||' for logical OR".to_string(),
                        ))
                    }
                }
                '\'' | '"' => self.read_string(),
                '0'..='9' => self.read_number(),
                'a'..='z' | 'A'..='Z' | '_' => self.read_identifier(),
                _ => Err(Error::InvalidValue(format!("Unexpected character: {}", ch))),
            },
        }
    }

    /// Skip whitespace characters
    fn skip_whitespace(&mut self) {
        while let Some(&ch) = self.chars.peek() {
            if ch.is_whitespace() {
                self.chars.next();
            } else {
                break;
            }
        }
    }

    /// Read a string literal
    fn read_string(&mut self) -> Result<Token> {
        let quote = self.chars.next().unwrap(); // consume opening quote
        let mut value = String::new();

        while let Some(ch) = self.chars.next() {
            if ch == quote {
                return Ok(Token::String(value));
            } else if ch == '\\' {
                // Handle escape sequences
                if let Some(escaped) = self.chars.next() {
                    match escaped {
                        'n' => value.push('\n'),
                        't' => value.push('\t'),
                        'r' => value.push('\r'),
                        '\\' => value.push('\\'),
                        '\'' => value.push('\''),
                        '"' => value.push('"'),
                        _ => {
                            value.push('\\');
                            value.push(escaped);
                        }
                    }
                }
            } else {
                value.push(ch);
            }
        }

        Err(Error::InvalidValue(
            "Unterminated string literal".to_string(),
        ))
    }

    /// Read a number literal
    fn read_number(&mut self) -> Result<Token> {
        let mut number = String::new();

        while let Some(&ch) = self.chars.peek() {
            if ch.is_ascii_digit() || ch == '.' {
                number.push(ch);
                self.chars.next();
            } else {
                break;
            }
        }

        match number.parse::<f64>() {
            Ok(value) => Ok(Token::Number(value)),
            Err(_) => Err(Error::InvalidValue(format!("Invalid number: {}", number))),
        }
    }

    /// Read an identifier or keyword
    fn read_identifier(&mut self) -> Result<Token> {
        let mut identifier = String::new();

        while let Some(&ch) = self.chars.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                identifier.push(ch);
                self.chars.next();
            } else {
                break;
            }
        }

        // Check for keywords
        match identifier.as_str() {
            "true" => Ok(Token::Boolean(true)),
            "false" => Ok(Token::Boolean(false)),
            "and" => Ok(Token::And),
            "or" => Ok(Token::Or),
            "not" => Ok(Token::Not),
            _ => {
                // Check if it's followed by '(' to determine if it's a function
                if self.chars.peek() == Some(&'(') {
                    Ok(Token::Function(identifier))
                } else {
                    Ok(Token::Identifier(identifier))
                }
            }
        }
    }
}

/// Parser for building expression AST
pub struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    /// Create a new parser with tokens
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            position: 0,
        }
    }

    /// Parse the tokens into an expression AST
    pub fn parse(&mut self) -> Result<Expr> {
        self.parse_or_expression()
    }

    /// Parse OR expressions
    fn parse_or_expression(&mut self) -> Result<Expr> {
        let mut left = self.parse_and_expression()?;

        while self.match_token(&Token::Or) {
            let op = BinaryOp::Or;
            let right = self.parse_and_expression()?;
            left = Expr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// Parse AND expressions
    fn parse_and_expression(&mut self) -> Result<Expr> {
        let mut left = self.parse_equality_expression()?;

        while self.match_token(&Token::And) {
            let op = BinaryOp::And;
            let right = self.parse_equality_expression()?;
            left = Expr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// Parse equality expressions (==, !=)
    fn parse_equality_expression(&mut self) -> Result<Expr> {
        let mut left = self.parse_comparison_expression()?;

        while let Some(op) = self.match_equality_operator() {
            let right = self.parse_comparison_expression()?;
            left = Expr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// Parse comparison expressions (<, <=, >, >=)
    fn parse_comparison_expression(&mut self) -> Result<Expr> {
        let mut left = self.parse_additive_expression()?;

        while let Some(op) = self.match_comparison_operator() {
            let right = self.parse_additive_expression()?;
            left = Expr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// Parse additive expressions (+, -)
    fn parse_additive_expression(&mut self) -> Result<Expr> {
        let mut left = self.parse_multiplicative_expression()?;

        while let Some(op) = self.match_additive_operator() {
            let right = self.parse_multiplicative_expression()?;
            left = Expr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// Parse multiplicative expressions (*, /, %)
    fn parse_multiplicative_expression(&mut self) -> Result<Expr> {
        let mut left = self.parse_power_expression()?;

        while let Some(op) = self.match_multiplicative_operator() {
            let right = self.parse_power_expression()?;
            left = Expr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// Parse power expressions (**)
    fn parse_power_expression(&mut self) -> Result<Expr> {
        let mut left = self.parse_unary_expression()?;

        if self.match_token(&Token::Power) {
            let right = self.parse_power_expression()?; // Right associative
            left = Expr::Binary {
                left: Box::new(left),
                op: BinaryOp::Power,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// Parse unary expressions (!, -, not)
    fn parse_unary_expression(&mut self) -> Result<Expr> {
        if self.match_token(&Token::Not) {
            let operand = self.parse_unary_expression()?;
            Ok(Expr::Unary {
                op: UnaryOp::Not,
                operand: Box::new(operand),
            })
        } else if self.match_token(&Token::Minus) {
            let operand = self.parse_unary_expression()?;
            Ok(Expr::Unary {
                op: UnaryOp::Negate,
                operand: Box::new(operand),
            })
        } else {
            self.parse_primary_expression()
        }
    }

    /// Parse primary expressions (literals, identifiers, function calls, parentheses)
    fn parse_primary_expression(&mut self) -> Result<Expr> {
        if let Some(token) = self.current_token().cloned() {
            match token {
                Token::Number(value) => {
                    self.advance();
                    Ok(Expr::Literal(LiteralValue::Number(value)))
                }
                Token::String(value) => {
                    self.advance();
                    Ok(Expr::Literal(LiteralValue::String(value)))
                }
                Token::Boolean(value) => {
                    self.advance();
                    Ok(Expr::Literal(LiteralValue::Boolean(value)))
                }
                Token::Identifier(name) => {
                    self.advance();
                    Ok(Expr::Column(name))
                }
                Token::Function(name) => {
                    let func_name = name;
                    self.advance();

                    if !self.match_token(&Token::LeftParen) {
                        return Err(Error::InvalidValue(
                            "Expected '(' after function name".to_string(),
                        ));
                    }

                    let mut args = Vec::new();

                    if !self.check_token(&Token::RightParen) {
                        loop {
                            args.push(self.parse_or_expression()?);

                            if !self.match_token(&Token::Comma) {
                                break;
                            }
                        }
                    }

                    if !self.match_token(&Token::RightParen) {
                        return Err(Error::InvalidValue(
                            "Expected ')' after function arguments".to_string(),
                        ));
                    }

                    Ok(Expr::Function {
                        name: func_name,
                        args,
                    })
                }
                Token::LeftParen => {
                    self.advance();
                    let expr = self.parse_or_expression()?;

                    if !self.match_token(&Token::RightParen) {
                        return Err(Error::InvalidValue(
                            "Expected ')' after expression".to_string(),
                        ));
                    }

                    Ok(expr)
                }
                _ => Err(Error::InvalidValue(format!(
                    "Unexpected token: {:?}",
                    token
                ))),
            }
        } else {
            Err(Error::InvalidValue("Unexpected end of input".to_string()))
        }
    }

    /// Helper methods for parsing
    fn current_token(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }

    fn advance(&mut self) {
        if self.position < self.tokens.len() {
            self.position += 1;
        }
    }

    fn match_token(&mut self, expected: &Token) -> bool {
        if self.check_token(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn check_token(&self, expected: &Token) -> bool {
        if let Some(token) = self.current_token() {
            std::mem::discriminant(token) == std::mem::discriminant(expected)
        } else {
            false
        }
    }

    fn match_equality_operator(&mut self) -> Option<BinaryOp> {
        match self.current_token() {
            Some(Token::Equal) => {
                self.advance();
                Some(BinaryOp::Equal)
            }
            Some(Token::NotEqual) => {
                self.advance();
                Some(BinaryOp::NotEqual)
            }
            _ => None,
        }
    }

    fn match_comparison_operator(&mut self) -> Option<BinaryOp> {
        match self.current_token() {
            Some(Token::LessThan) => {
                self.advance();
                Some(BinaryOp::LessThan)
            }
            Some(Token::LessThanOrEqual) => {
                self.advance();
                Some(BinaryOp::LessThanOrEqual)
            }
            Some(Token::GreaterThan) => {
                self.advance();
                Some(BinaryOp::GreaterThan)
            }
            Some(Token::GreaterThanOrEqual) => {
                self.advance();
                Some(BinaryOp::GreaterThanOrEqual)
            }
            _ => None,
        }
    }

    fn match_additive_operator(&mut self) -> Option<BinaryOp> {
        match self.current_token() {
            Some(Token::Plus) => {
                self.advance();
                Some(BinaryOp::Add)
            }
            Some(Token::Minus) => {
                self.advance();
                Some(BinaryOp::Subtract)
            }
            _ => None,
        }
    }

    fn match_multiplicative_operator(&mut self) -> Option<BinaryOp> {
        match self.current_token() {
            Some(Token::Multiply) => {
                self.advance();
                Some(BinaryOp::Multiply)
            }
            Some(Token::Divide) => {
                self.advance();
                Some(BinaryOp::Divide)
            }
            Some(Token::Modulo) => {
                self.advance();
                Some(BinaryOp::Modulo)
            }
            _ => None,
        }
    }
}
