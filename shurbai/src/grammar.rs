use std::str::FromStr;

use llama_cpp_2::grammar::LlamaGrammar;


pub const JSON_GRAMMAR:&str = "root   ::= object\r\nvalue  ::= object | array | string | number | (\"true\" | \"false\" | \"null\")\r\n\r\nobject ::=\r\n  \"{\" (\r\n            string \":\" value\r\n    (\",\" string \":\" value)*\r\n  )? \"}\" \r\n\r\narray  ::=\r\n  \"[\" (\r\n            value\r\n    (\",\" value)*\r\n  )? \"]\" \r\n\r\nstring ::=\r\n  \"\\\"\" (\r\n    [^\"\\\\] |\r\n    \"\\\\\" ([\"\\\\/bfnrt] | \"u\" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes\r\n  )* \"\\\"\"\r\n\r\nnumber ::= (\"-\"? ([0-9] | [1-9] [0-9]*)) (\".\" [0-9]+)? ([eE] [-+]? [0-9]+)?";


pub fn load_grammar() -> LlamaGrammar {
    LlamaGrammar::from_str(JSON_GRAMMAR).expect("Failed to load grammar")
}