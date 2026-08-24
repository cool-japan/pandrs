//! Lexical analysis and parsing for query expressions
//!
//! This module provides lexical analysis (tokenization) and parsing functionality
//! to convert query strings into abstract syntax trees (AST).

use std::iter::Peekable;
use std::str::Chars;

use super::ast::{BinaryOp, Expr, LiteralValue, Token, UnaryOp};
use crate::core::error::{Error, Result};

/// Lexer for tokenizing query expressions.
///
/// The lexer borrows the query string for the lifetime `'a`; no `'static`
/// promotion (and therefore no `transmute`) is required to build one from a
/// short-lived `&str`.
pub struct Lexer<'a> {
    chars: Peekable<Chars<'a>>,
    /// The original source, retained for diagnostics.
    input: &'a str,
}

impl<'a> Lexer<'a> {
    /// Create a new lexer
    pub fn new(input: &'a str) -> Self {
        Self {
            chars: input.chars().peekable(),
            input,
        }
    }

    /// The query string this lexer was built from.
    pub fn input(&self) -> &'a str {
        self.input
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
                // pandas spells logical AND as a single '&'; '&&' is also
                // accepted for the Rust-flavoured syntax this crate started
                // with.
                '&' => {
                    self.chars.next();
                    if self.chars.peek() == Some(&'&') {
                        self.chars.next();
                    }
                    Ok(Token::And)
                }
                // Likewise '|' and '||' are both logical OR.
                '|' => {
                    self.chars.next();
                    if self.chars.peek() == Some(&'|') {
                        self.chars.next();
                    }
                    Ok(Token::Or)
                }
                // pandas spells logical NOT as '~'.
                '~' => {
                    self.chars.next();
                    Ok(Token::Not)
                }
                // '@name' explicitly references a query-context variable.
                '@' => {
                    self.chars.next();
                    let name = self.read_word();
                    if name.is_empty() {
                        Err(Error::InvalidValue(
                            "Expected a variable name after '@'".to_string(),
                        ))
                    } else {
                        Ok(Token::Variable(name))
                    }
                }
                // Backtick-quoted identifiers allow column names containing
                // spaces or punctuation, as in pandas.
                '`' => self.read_quoted_identifier(),
                '\'' | '"' => self.read_string(),
                '0'..='9' => self.read_number(),
                '.' => self.read_number(),
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
        let quote = self.chars.next().ok_or_else(|| {
            Error::InvalidInput("Expected quote character for string literal".to_string())
        })?; // consume opening quote
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

    /// Read a number literal, including `1.5`, `.5` and `1e-3` forms.
    fn read_number(&mut self) -> Result<Token> {
        let mut number = String::new();
        let mut seen_dot = false;

        while let Some(&ch) = self.chars.peek() {
            if ch.is_ascii_digit() {
                number.push(ch);
                self.chars.next();
            } else if ch == '.' && !seen_dot {
                seen_dot = true;
                number.push(ch);
                self.chars.next();
            } else if (ch == 'e' || ch == 'E') && !number.is_empty() {
                // Only consume the exponent when it is actually well formed,
                // so an identifier such as `e_id` is not swallowed.
                let mut lookahead = self.chars.clone();
                lookahead.next();
                let mut exponent = String::from("e");
                if let Some(&sign) = lookahead.peek() {
                    if sign == '+' || sign == '-' {
                        exponent.push(sign);
                        lookahead.next();
                    }
                }
                let mut digits = 0usize;
                while let Some(&digit) = lookahead.peek() {
                    if digit.is_ascii_digit() {
                        exponent.push(digit);
                        lookahead.next();
                        digits += 1;
                    } else {
                        break;
                    }
                }
                if digits == 0 {
                    break;
                }
                number.push_str(&exponent);
                self.chars = lookahead;
            } else {
                break;
            }
        }

        match number.parse::<f64>() {
            Ok(value) => Ok(Token::Number(value)),
            Err(_) => Err(Error::InvalidValue(format!("Invalid number: {}", number))),
        }
    }

    /// Read a bare word (identifier characters only).
    fn read_word(&mut self) -> String {
        let mut word = String::new();

        while let Some(&ch) = self.chars.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                word.push(ch);
                self.chars.next();
            } else {
                break;
            }
        }

        word
    }

    /// Read a backtick-quoted identifier, which may contain any character
    /// except the closing backtick.
    fn read_quoted_identifier(&mut self) -> Result<Token> {
        self.chars.next(); // consume the opening backtick
        let mut identifier = String::new();

        for ch in self.chars.by_ref() {
            if ch == '`' {
                return Ok(Token::Identifier(identifier));
            }
            identifier.push(ch);
        }

        Err(Error::InvalidValue(
            "Unterminated backtick-quoted identifier".to_string(),
        ))
    }

    /// Read an identifier or keyword
    fn read_identifier(&mut self) -> Result<Token> {
        let identifier = self.read_word();

        // Check for keywords. Both the Python/pandas spellings (`True`,
        // `False`, `and`, `or`, `not`) and the lowercase Rust spellings are
        // accepted.
        match identifier.as_str() {
            "true" | "True" => Ok(Token::Boolean(true)),
            "false" | "False" => Ok(Token::Boolean(false)),
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

    /// Parse the tokens into an expression AST.
    ///
    /// The whole token stream must be consumed: trailing tokens are a syntax
    /// error rather than being silently ignored (which would quietly change the
    /// meaning of a query).
    pub fn parse(&mut self) -> Result<Expr> {
        let expr = self.parse_or_expression()?;

        match self.current_token() {
            None | Some(Token::Eof) => Ok(expr),
            Some(token) => Err(Error::InvalidValue(format!(
                "Unexpected trailing token in query expression: {:?}",
                token
            ))),
        }
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
                    // `name (args)` with whitespace before the parenthesis is a
                    // function call too; the lexer only tags a name as a
                    // function when '(' follows immediately.
                    if self.check_token(&Token::LeftParen) {
                        return self.parse_call_arguments(name);
                    }
                    Ok(Expr::Column(name))
                }
                Token::Variable(name) => {
                    self.advance();
                    Ok(Expr::Variable(name))
                }
                Token::Function(name) => {
                    self.advance();
                    self.parse_call_arguments(name)
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

    /// Parse the argument list of a function call, starting at '('.
    fn parse_call_arguments(&mut self, name: String) -> Result<Expr> {
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

        Ok(Expr::Function { name, args })
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
