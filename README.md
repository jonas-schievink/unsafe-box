# A more convenient way of dealing with raw pointers

[![crates.io](https://img.shields.io/crates/v/unsafe-box.svg)](https://crates.io/crates/unsafe-box)
[![docs.rs](https://docs.rs/unsafe-box/badge.svg)](https://docs.rs/unsafe-box/)
[![Build Status](https://travis-ci.org/jonas-schievink/unsafe-box.svg?branch=master)](https://travis-ci.org/jonas-schievink/unsafe-box)

This crate provides a more explicit and convenient way of storing multiple raw
pointers pointing to the same object. This maps well to a common C pattern,
where a data structure sometimes stores multiple pointers to some object and
manages its lifetime.

While the type exported by this crate are similar to raw pointers and implement
mostly `unsafe` methods where the user must validate that operations are safe,
all types will employ additional checks in debug mode that can catch many
mistakes (such as freeing an object while there are still pointers to it, or
obtaining conflicting mutable/immutable references to the object).

It can also be used as a much simpler (but unsafe) alternative to the great
[`rental`](https://github.com/jpernst/rental) crate (depending on your needs).

Check out the [crate documentation](https://docs.rs/unsafe-box/) for more info,
and see [`examples/multi-lookup-storage.rs`](examples/multi-lookup-storage.rs)
for an example showing how to build a safe data structure with this crate.

Please refer to the [changelog](CHANGELOG.md) to see what changed in the last
releases.

## Usage

Start by adding an entry to your `Cargo.toml`:

```toml
[dependencies]
unsafe-box = "0.1.0"
```

Then import the crate into your Rust code and `use` the types you need:

```rust
extern crate unsafe_box;

use unsafe_box::{UnsafeBox, UnsafeRef};
```
