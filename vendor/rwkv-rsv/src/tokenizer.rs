//! RWKV 分词器（从 web-rwkv 移植，去掉 wasm/ahash 依赖，改用 std 集合）。
//!
//! 词表格式与 web-rwkv 一致：一个 `{token_id: 字符串或字节数组}` 的 JSON 映射，
//! 例如官方的 `rwkv_vocab_v20230424.json`。编码采用最长匹配贪心。

use std::collections::{BTreeMap, HashMap};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("failed to parse vocabulary: {0}")]
    FailedToParseVocabulary(serde_json::Error),
    #[error("no matching token found")]
    NoMatchingTokenFound,
    #[error("out of range token: {0}")]
    OutOfRangeToken(u32),
}

#[derive(Debug, Clone)]
pub struct Tokenizer {
    first_bytes_to_lengths: Vec<Box<[u16]>>,
    bytes_to_token_index: HashMap<Vec<u8>, u32>,
    token_index_to_bytes: Vec<Vec<u8>>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
enum StrOrBytes {
    Str(String),
    Bytes(Vec<u8>),
}

impl Tokenizer {
    pub fn new(vocab: &str) -> Result<Self, TokenizerError> {
        let map: BTreeMap<u32, StrOrBytes> =
            serde_json::from_str(vocab).map_err(TokenizerError::FailedToParseVocabulary)?;

        let list: Vec<(Vec<u8>, u32)> = map
            .into_iter()
            .map(|(token, pattern)| {
                let pattern: Vec<u8> = match pattern {
                    StrOrBytes::Str(string) => string.into_bytes(),
                    StrOrBytes::Bytes(bytes) => bytes,
                };
                (pattern, token)
            })
            .collect();

        let mut first_bytes_to_lengths = Vec::new();
        first_bytes_to_lengths.resize(u16::MAX as usize, {
            let mut set = std::collections::BTreeSet::new();
            set.insert(1);
            set
        });

        let mut token_index_to_bytes = Vec::new();
        let max_token_index = list.iter().map(|(_, index)| *index).max().unwrap_or(0) as usize;
        token_index_to_bytes.resize_with(max_token_index + 1, Vec::new);

        let mut bytes_to_token_index = HashMap::new();
        for (token_bytes, token_index) in list {
            if token_bytes.len() >= 2 {
                let key = u16::from_ne_bytes([token_bytes[0], token_bytes[1]]) as usize;
                first_bytes_to_lengths[key].insert(token_bytes.len() as u16);
            }

            bytes_to_token_index.insert(token_bytes.clone(), token_index);
            token_index_to_bytes[token_index as usize] = token_bytes;
        }

        let first_bytes_to_lengths: Vec<Box<[_]>> = first_bytes_to_lengths
            .into_iter()
            .map(|inner| {
                let mut inner: Vec<_> = inner.into_iter().collect();
                inner.sort_unstable_by_key(|l| !*l);
                inner.into_boxed_slice()
            })
            .collect();

        Ok(Tokenizer {
            first_bytes_to_lengths,
            bytes_to_token_index,
            token_index_to_bytes,
        })
    }

    pub fn encode(&self, input: &[u8]) -> Result<Vec<u32>, TokenizerError> {
        let mut output = Vec::new();
        self.encode_into(input, &mut output)?;
        Ok(output)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<Vec<u8>, TokenizerError> {
        let mut output = Vec::with_capacity(tokens.len());
        self.decode_into(tokens, &mut output)?;
        Ok(output)
    }

    pub fn encode_into(
        &self,
        mut input: &[u8],
        output: &mut Vec<u32>,
    ) -> Result<(), TokenizerError> {
        'next_token: while !input.is_empty() {
            let lengths = if input.len() >= 2 {
                let key = u16::from_ne_bytes([input[0], input[1]]) as usize;
                &self.first_bytes_to_lengths[key][..]
            } else {
                &[1][..]
            };

            for &length in lengths {
                let length = length as usize;
                if length > input.len() {
                    continue;
                }

                if let Some(&token_index) = self.bytes_to_token_index.get(&input[..length]) {
                    output.push(token_index);
                    input = &input[length..];
                    continue 'next_token;
                }
            }

            return Err(TokenizerError::NoMatchingTokenFound);
        }

        Ok(())
    }

    pub fn decode_into(&self, tokens: &[u32], output: &mut Vec<u8>) -> Result<(), TokenizerError> {
        for &token in tokens {
            let bytes = self
                .token_index_to_bytes
                .get(token as usize)
                .ok_or(TokenizerError::OutOfRangeToken(token))?;

            output.extend_from_slice(bytes);
        }

        Ok(())
    }
}
