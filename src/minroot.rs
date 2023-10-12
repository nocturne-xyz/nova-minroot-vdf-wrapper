// adapted from the example in the main nova repo
// almost exactly the same; main difference 
// is that the outputs of the VDF are PIs so the verifier can check against a claimed output/input pair

//! Demonstrates how to use Nova to produce a recursive proof of the correct execution of
//! iterations of the `MinRoot` function, thereby realizing a Nova-based verifiable delay function (VDF).
//! We execute a configurable number of iterations of the `MinRoot` function per step of Nova's recursion.
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::PrimeField;
use nova_snark::traits::circuit::StepCircuit;
use num_bigint::BigUint;

#[derive(Clone, Debug)]
pub struct MinRootIteration<F: PrimeField> {
    pub x_i: F,
    pub y_i: F,
    pub x_i_plus_1: F,
    pub y_i_plus_1: F,
}

impl<F: PrimeField> MinRootIteration<F> {
    // produces a sample non-deterministic advice, executing one invocation of MinRoot per step
    pub fn new(num_iters: usize, x_0: &F, y_0: &F) -> (Vec<F>, Vec<Self>) {
        // although this code is written generically, it is tailored to Pallas' scalar field
        // (p - 3 / 5)
        let exp = BigUint::parse_bytes(
            b"23158417847463239084714197001737581570690445185553317903743794198714690358477",
            10,
        )
        .unwrap();

        let mut res = Vec::new();
        let mut x_i = *x_0;
        let mut y_i = *y_0;
        for _i in 0..num_iters {
            let x_i_plus_1 = (x_i + y_i).pow_vartime(exp.to_u64_digits()); // computes the fifth root of x_i + y_i

            // sanity check
            let sq = x_i_plus_1 * x_i_plus_1;
            let quad = sq * sq;
            let fifth = quad * x_i_plus_1;
            debug_assert_eq!(fifth, x_i + y_i);

            let y_i_plus_1 = x_i;

            res.push(Self {
                x_i,
                y_i,
                x_i_plus_1,
                y_i_plus_1,
            });

            x_i = x_i_plus_1;
            y_i = y_i_plus_1;
        }

        let z0 = vec![*x_0, *y_0];

        (z0, res)
    }
}

#[derive(Clone, Debug)]
pub struct MinRootCircuit<F: PrimeField> {
    pub(crate) seq: Vec<MinRootIteration<F>>,
}

impl<F> StepCircuit<F> for MinRootCircuit<F>
where
    F: PrimeField,
{
    fn arity(&self) -> usize {
        2
    }

    fn synthesize<CS: ConstraintSystem<F>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
        let mut z_out: Result<Vec<AllocatedNum<F>>, SynthesisError> =
            Err(SynthesisError::AssignmentMissing);

        // use the provided inputs
        let x_0 = z[0].clone();
        let y_0 = z[1].clone();

        // variables to hold running x_i and y_i
        let mut x_i = x_0;
        let mut y_i = y_0;
        for i in 0..self.seq.len() {
            // non deterministic advice
            let x_i_plus_1 =
                AllocatedNum::alloc(cs.namespace(|| format!("x_i_plus_1_iter_{i}")), || {
                    Ok(self.seq[i].x_i_plus_1)
                })?;

            // check the following conditions hold:
            // (i) x_i_plus_1 = (x_i + y_i)^{1/5}, which can be more easily checked with x_i_plus_1^5 = x_i + y_i
            // (ii) y_i_plus_1 = x_i
            // (1) constraints for condition (i) are below
            // (2) constraints for condition (ii) is avoided because we just used x_i wherever y_i_plus_1 is used
            let x_i_plus_1_sq =
                x_i_plus_1.square(cs.namespace(|| format!("x_i_plus_1_sq_iter_{i}")))?;
            let x_i_plus_1_quad =
                x_i_plus_1_sq.square(cs.namespace(|| format!("x_i_plus_1_quad_{i}")))?;
            cs.enforce(
                || format!("x_i_plus_1_quad * x_i_plus_1 = x_i + y_i_iter_{i}"),
                |lc| lc + x_i_plus_1_quad.get_variable(),
                |lc| lc + x_i_plus_1.get_variable(),
                |lc| lc + x_i.get_variable() + y_i.get_variable(),
            );

            if i == self.seq.len() - 1 {
                z_out = Ok(vec![x_i_plus_1.clone(), x_i.clone()]);
            }

            // update x_i and y_i for the next iteration
            y_i = x_i;
            x_i = x_i_plus_1;
        }

        z_out
    }
}
