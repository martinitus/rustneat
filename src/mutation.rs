use gene::Gene;

pub trait Mutation {}

impl dyn Mutation {
    pub fn connection_weight(gene: &mut Gene, perturbation: bool) {
        let mut new_weight = Gene::generate_weight();
        if perturbation {
            new_weight += gene.weight;
        }
        gene.weight = new_weight;
    }

    pub fn add_connection(in_neuron_id: usize, out_neuron_id: usize) -> Gene {
        Gene::new(in_neuron_id, out_neuron_id)
    }

    pub fn add_neuron(gene: &mut Gene, new_neuron_id: usize) -> (Gene, Gene) {
        gene.disable();

        let gen1 = Gene { source_id: gene.source_id, target_id: new_neuron_id, weight: 1f64, enabled: true, bias: false };
        let gen2 = Gene { source_id: new_neuron_id, target_id: gene.target_id, weight: gene.weight, enabled: true, bias: false };

        (gen1, gen2)
    }

    pub fn toggle_expression(gene: &mut Gene) {
        gene.toggle_enabled();
    }

    pub fn toggle_bias(gene: &mut Gene) {
        gene.toggle_bias();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gene::Gene;

    #[test]
    fn mutate_toggle_gene_should_toggle() {
        let mut gene = Gene { source_id: 0, target_id: 1, weight: 1f64, enabled: false, bias: false };

        <dyn Mutation>::toggle_expression(&mut gene);
        assert!(gene.enabled);

        <dyn Mutation>::toggle_expression(&mut gene);
        assert!(!gene.enabled);
    }
}
