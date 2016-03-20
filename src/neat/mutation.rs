use neat::connection_gene::ConnectionGene as ConnectionGene;
use neat::genome::Genome as Genome;

pub trait Mutation {
}

impl Mutation {
    pub fn connection_weight (gene: &mut ConnectionGene) {
        gene.weight = ConnectionGene::generate_weight()
    }

    pub fn add_connection (in_node_id: u32, out_node_id: u32, genome: &mut Genome) -> (ConnectionGene) {
        genome.global_innovation += 1;
        ConnectionGene { 
            in_node_id: in_node_id,
            out_node_id: out_node_id,
            innovation: genome.global_innovation,
            ..Default::default()
        } 
    }

    pub fn add_node (gene: &mut ConnectionGene, new_node_id: u32, genome: &mut Genome) -> (ConnectionGene, ConnectionGene) {
        gene.enabled = false;

        genome.global_innovation += 1;
        let gen1 = ConnectionGene {
            in_node_id: gene.in_node_id,
            out_node_id: new_node_id,
            weight: 1f64,
            innovation: genome.global_innovation,
            ..Default::default()
        };

        genome.global_innovation += 1;
        let gen2 = ConnectionGene {
            in_node_id: new_node_id,
            out_node_id: gene.out_node_id,
            weight: gene.weight,
            innovation: genome.global_innovation,
            ..Default::default()
        };

        (gen1, gen2)
    }
}

#[test]
fn mutation_connection_weight(){
    let mut genome = Genome::new();
    let mut gene = genome.create_gene();
    let orig_gene = gene.clone();
    Mutation::connection_weight(&mut gene);

    assert!(gene.weight != orig_gene.weight);
}

#[test]
fn mutation_add_connection(){
    let mut genome = Genome::new();
    let new_gene = Mutation::add_connection(1, 2, &mut genome);

    assert!(new_gene.in_node_id == 1);
    assert!(new_gene.out_node_id == 2);
    assert!(new_gene.innovation == 1);
}

#[test]
fn mutation_add_node(){
    let mut genome = Genome::new();
    let mut gene = genome.create_gene();
    let (new_gene1, new_gene2) = Mutation::add_node(&mut gene, 3, &mut genome);

    assert!(!gene.enabled);
    assert!(new_gene1.in_node_id == gene.in_node_id);
    assert!(new_gene1.out_node_id == 3);
    assert!(new_gene2.in_node_id == 3);
    assert!(new_gene2.out_node_id == gene.out_node_id);
    assert!(new_gene1.innovation == 2);
    assert!(new_gene2.innovation == 3);
}
