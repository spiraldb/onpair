// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::BuildHasherDefault;
use std::hash::Hasher;

const K: u64 = 0x517c_c1b7_2722_0a95;

#[derive(Default)]
pub struct FxHasher {
    hash: u64,
}

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.hash = (self.hash.rotate_left(5) ^ b as u64).wrapping_mul(K);
        }
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.hash = (self.hash.rotate_left(5) ^ i as u64).wrapping_mul(K);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.hash = (self.hash.rotate_left(5) ^ i).wrapping_mul(K);
    }
}

pub type FxBuildHasher = BuildHasherDefault<FxHasher>;

pub type MapHasher = hashbrown::DefaultHashBuilder;

/// LPM map type, parameterised over the feature-selected [`MapHasher`].
pub type Map<K, V> = hashbrown::HashMap<K, V, MapHasher>;

/// An empty [`Map`] with the selected hasher.
#[inline]
pub fn map<K, V>() -> Map<K, V> {
    Map::with_hasher(MapHasher::default())
}

/// A [`Map`] preallocated for `cap` entries with the selected hasher.
#[inline]
pub fn map_with_capacity<K, V>(cap: usize) -> Map<K, V> {
    Map::with_capacity_and_hasher(cap, MapHasher::default())
}
