//! Implementation of `NeuroEvolution` of Augmenting Topologies [NEAT]
//! (http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
#[macro_use]
extern crate generator;
extern crate rand;
extern crate approx;
extern crate ndarray;
extern crate lazycell;
extern crate petgraph;

pub mod population;
pub mod compatibility;
pub mod evolution;
