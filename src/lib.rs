//! Implementation of `NeuroEvolution` of Augmenting Topologies [NEAT]
//! (http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
#[macro_use]
extern crate generator;
extern crate rand;
extern crate approx;
extern crate ndarray;
extern crate lazycell;
extern crate petgraph;

pub use self::population::Population;

pub mod population;
