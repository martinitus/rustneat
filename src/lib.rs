#![deny(
missing_docs,
trivial_casts,
trivial_numeric_casts,
unsafe_code,
unused_import_braces,
unused_qualifications
)]
#![cfg_attr(feature = "clippy", feature(plugin))]
#![cfg_attr(feature = "clippy", plugin(clippy))]
#![cfg_attr(
feature = "clippy",
deny(clippy, unicode_not_nfc, wrong_pub_self_convention)
)]
#![cfg_attr(feature = "clippy", allow(use_debug, too_many_arguments))]

//! Implementation of `NeuroEvolution` of Augmenting Topologies [NEAT]
//! (http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)

extern crate rand;
extern crate approx;
extern crate ndarray;
extern crate lazycell;

pub use self::population::Population;

pub mod population;
