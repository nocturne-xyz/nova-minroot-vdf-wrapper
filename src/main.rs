use clap::Parser;
use nova_snark::{
    traits::{circuit::TrivialCircuit, Group},
    CompressedSNARK, PublicParams, RecursiveSNARK,
};
use std::{iter, mem, sync::mpsc::sync_channel, thread, time::Instant};

mod minroot;
use minroot::*;

type G1 = pasta_curves::pallas::Point;
type G2 = pasta_curves::vesta::Point;
type F = <G1 as Group>::Scalar;
type F2 = <G2 as Group>::Scalar;

/// args
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// input to the VDF as a hex string.
    /// will be interpreted as a byte array and hashed into a pair of field elements
    /// before being fed into the VDF
    #[arg(short, long)]
    input: String,

    /// Log base 2 of the number of iterations of MinRoot we'd like to execute.
    /// The paper estimates 2^30 iterations of MinRoot corresponds to a delay of ~1 second,
    /// therefore it reccomends the following settings:
    /// - 2^36 for ~1 minute
    /// - 2^42 for ~1 hour
    /// - 2^46 for ~1 day
    #[arg(short, long)]
    log_num_iters: usize,

    /// Log base 2 of number of iterations of MinRoot per folded "step" circuit instance
    /// Defaults to 16.
    ///
    /// If you're running with the `cuda` feature enabled,
    /// you'll probably want to bump this up
    #[arg(short, long, default_value_t = 16)]
    log_num_iters_per_step: usize,

    /// Path to write output of the VDF to
    /// defaults to "./output.bin"
    #[arg(short, long, default_value_t=String::from("./output.bin"))]
    output_path: String,

    /// Path to write serialized proof to
    /// defaults to "./proof.bin"
    #[arg(short, long, default_value_t=String::from("./proof.bin"))]
    proof_path: String,
}

fn main() {
    let args = Args::parse();

    let num_iters_per_step = 1 << args.log_num_iters_per_step;
    let num_steps = 1 << (args.log_num_iters - args.log_num_iters_per_step);

    // prepare input
    let _input_bytes = if args.input.starts_with("0x") {
        hex::decode(&args.input[2..]).unwrap()
    } else {
        hex::decode(&args.input).unwrap()
    };

    // TODO - hash input_bytes into a pair of field elements
    let (x0, y0) = (F::zero(), F::one());

    println!(
        "Running MinRoot VDF for 2^{} iterations",
        args.log_num_iters
    );

    // the rest of this function's body is adapted from main() in https://github.com/microsoft/Nova/blob/main/examples/minroot.rs

    let circuit_primary = MinRootCircuit {
        seq: vec![
            MinRootIteration {
                x_i: F::zero(),
                y_i: F::zero(),
                x_i_plus_1: F::zero(),
                y_i_plus_1: F::zero(),
            };
            num_iters_per_step
        ],
    };
    let circuit_secondary = TrivialCircuit::default();

    // produce public parameters
    let start = Instant::now();
    println!("Producing public parameters...");
    let pp = PublicParams::<G1, G2, MinRootCircuit<F>, TrivialCircuit<F2>>::setup(
        &circuit_primary,
        &circuit_secondary,
    );
    println!("PublicParams::setup, took {:?} ", start.elapsed());

    println!(
        "Number of constraints per step (primary circuit): {}",
        pp.num_constraints().0
    );
    println!(
        "Number of constraints per step (secondary circuit): {}",
        pp.num_constraints().1
    );

    println!(
        "Number of variables per step (primary circuit): {}",
        pp.num_variables().0
    );
    println!(
        "Number of variables per step (secondary circuit): {}",
        pp.num_variables().1
    );

    // proceed with two threads - one that runs the VDF / generates witnesses, the other for proving

    let (tx, rx) = sync_channel::<MinRootCircuit<F>>(32);
    // start witness generation thread
    let witness_thread = thread::spawn(move || {
        let instances = (0..num_steps).scan((x0, y0), |(x0, y0), _| {
            let seq = MinRootIteration::new(num_iters_per_step, x0, y0).1;
            *x0 = seq.last().unwrap().x_i_plus_1;
            *y0 = seq.last().unwrap().y_i_plus_1;
            Some(MinRootCircuit { seq })
        });

        for instance in instances {
            tx.send(instance).unwrap();
        }
    });

    // initial PIs
    let z0_primary = vec![x0, y0];
    let z0_secondary = vec![F2::zero()];

    let mut instances = rx.into_iter();

    // get the first instance out and use it as the base case
    let first_instance = instances.next().unwrap();
    let mut folded_instance = RecursiveSNARK::<G1, G2, MinRootCircuit<F>, TrivialCircuit<F2>>::new(
        &pp,
        &first_instance,
        &circuit_secondary,
        z0_primary.clone(),
        z0_secondary.clone(),
    );

    // iterate over them all and prove
    let mut output = (x0, y0);
    for (i, instance) in iter::once(first_instance).chain(instances).enumerate() {
        println!("Proving step {}", i);
        folded_instance
            .prove_step(
                &pp,
                &instance,
                &circuit_secondary,
                z0_primary.clone(),
                z0_secondary.clone(),
            )
            .unwrap_or_else(|e| panic!("failed to prove step {} due to error: {:#?}", i, e));

        let last_iteration = instance.seq.last().unwrap();
        output = (last_iteration.x_i_plus_1, last_iteration.y_i_plus_1);
    }

    // wait for witness generation to finish
    witness_thread.join().unwrap();

    println!("Verifying the folded instance...");
    // verify the folded_instance and check output matches what we got in witness gen
    let (zn, _) = folded_instance
        .verify(&pp, num_steps, &z0_primary, &z0_secondary)
        .expect("folded instance invalid");
    assert_eq!((zn[0], zn[1]), output);
    println!("Folded instance verifier accepts, output matches witness gen");

    // compress the folded instance in a wrapper snark proof
    println!("Compressing the folded instance with a wrapper SNARK");
    type EE1 = nova_snark::provider::ipa_pc::EvaluationEngine<G1>;
    type EE2 = nova_snark::provider::ipa_pc::EvaluationEngine<G2>;
    type S1 = nova_snark::spartan::snark::RelaxedR1CSSNARK<G1, EE1>;
    type S2 = nova_snark::spartan::snark::RelaxedR1CSSNARK<G2, EE2>;

    let (pk, vk) = CompressedSNARK::<_, _, _, _, S1, S2>::setup(&pp).unwrap();
    let proof = CompressedSNARK::<_, _, _, _, S1, S2>::prove(&pp, &pk, &folded_instance)
        .expect("failed to prove wrapper snark");

    println!("Verifying the wrapper SNARK proof...");
    // verify proof, check that the output matches what we got in witness gen
    let (zn, _) = proof
        .verify(&vk, num_steps, z0_primary, z0_secondary)
        .expect("wrapper snark proof invalid");
    assert_eq!((zn[0], zn[1]), output);
    println!("Wrapper SNARK verifier accepts, output matches witness gen");

    // TODO encode output to a hex string
    println!("VDF Output: {:?}", output);

    // TODO write proof to a file
}
