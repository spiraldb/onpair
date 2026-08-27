// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Encode side: FSST compresses, then the table is sorted and the stream rewritten.

use fsst::Compressor;

use crate::core::dictionary::pad_raw;
use crate::{CompactDictionary, InvalidColumn, MAX_TOKEN_SIZE, OwnedDictionaryStorage};

const MAX_SYMBOL_SIZE: usize = size_of::<u64>();

/// Where each symbol landed in `sorted`, indexed by its FSST code. A miss means the
/// two were built from different symbol tables.
fn code_map(tokens: &[&[u8]], sorted: &[&[u8]]) -> Result<Vec<u8>, InvalidColumn> {
    tokens
        .iter()
        .map(|token| {
            let id = sorted
                .binary_search(token)
                .map_err(|_| InvalidColumn::CodeOutOfRange)?;
            u8::try_from(id).map_err(|_| InvalidColumn::CodeOutOfRange)
        })
        .collect()
}

/// Remap symbol codes to token ids in place. An escape marker and the literal
/// after it both pass through.
fn remap_codes(codes: &[u8], code_map: &[u8]) -> Result<Vec<u8>, InvalidColumn> {
    let mut out = codes.to_vec();
    let mut i = 0;
    while let Some(&code) = out.get(i) {
        if code == fsst::ESCAPE_CODE {
            i += 2;
        } else {
            out[i] = *code_map
                .get(code as usize)
                .ok_or(InvalidColumn::CodeOutOfRange)?;
            i += 1;
        }
    }
    Ok(out)
}

/// Sort an FSST symbol table into a safety-validated dictionary and rewrite `codes`
/// to index it. `codes` must come from `compressor`.
pub fn transcode_onpair(
    compressor: &Compressor,
    codes: &[u8],
) -> Result<(CompactDictionary, Vec<u8>), InvalidColumn> {
    // A symbol is a `u64` plus a length, and `symbol_lengths` is the authority:
    // `Symbol::len` undercounts a symbol ending in `0x00`.
    let raw: Vec<[u8; MAX_SYMBOL_SIZE]> = compressor
        .symbol_table()
        .iter()
        .map(|symbol| symbol.to_u64().to_le_bytes())
        .collect();
    let tokens: Vec<&[u8]> = raw
        .iter()
        .zip(compressor.symbol_lengths())
        .map(|(bytes, &len)| match len as usize {
            0 => Err(InvalidColumn::EmptyToken),
            len if len > MAX_SYMBOL_SIZE => Err(InvalidColumn::TokenTooLarge),
            len => Ok(&bytes[..len]),
        })
        .collect::<Result<_, _>>()?;

    // Deduplicated: a repeated symbol would break strictly-ascending offsets, so both
    // codes then map to the one token.
    let mut sorted = tokens.clone();
    sorted.sort_unstable();
    sorted.dedup();

    let code_map = code_map(&tokens, &sorted)?;

    let mut bytes = Vec::with_capacity(sorted.len() * MAX_SYMBOL_SIZE + MAX_TOKEN_SIZE);
    let mut offsets = Vec::with_capacity(sorted.len() + 1);
    offsets.push(0);
    for token in &sorted {
        bytes.extend_from_slice(token);
        offsets.push(bytes.len() as u32);
    }
    pad_raw(&mut bytes, &offsets);

    let dictionary =
        CompactDictionary::validate_safety(OwnedDictionaryStorage::new(bytes, offsets))?;
    Ok((dictionary, remap_codes(codes, &code_map)?))
}
