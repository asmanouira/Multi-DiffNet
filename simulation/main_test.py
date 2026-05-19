# examples/main_test.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from multi_diffnet import MultiDiffNet
from simulation import simulate_one_scenario


def main():
    data, truth = simulate_one_scenario(
        p=80,
        r=3,
        n0=20,
        n_low=20,
        n_high=20,
        seed=2016,
        verbose=True,
    )

    model = MultiDiffNet(
        lambda_1=0.1,
        lambda_2={
            "low": 0.6,
            "high": 0.6,
        },
        lambda_3=0.01,
        mu0=2.5,
        rho=1.0,
        eps=1e-2,
        maxiter_baseline=100,
        maxiter_admm=10,
        maxiter_ama=10,
        maxiter_outer=5,
        verbose=True,
    )

    fit = model.fit(data)

    out_dir = Path("results_one_simulated_scenario")
    model.save_results(out_dir, data)

    print("\n=== Fit summary ===")
    print(f"data_0 shape:       {data.data_0.shape}")
    print(f"data_K shape:       {data.data_K.shape}")
    print(f"groups:             {list(data.group_names)}")
    print(f"Theta0 shape:       {fit['estimate_0']['Theta_0'].shape}")
    print(f"Theta groups shape: {fit['estimate_K']['Theta'].shape}")
    print(f"Delta groups shape: {fit['estimate_K']['Delta'].shape}")
    print(f"Results saved in:   {out_dir.resolve()}")

    assert fit["estimate_0"]["Theta_0"].shape == truth["Theta0"].shape
    assert fit["estimate_K"]["Theta"].shape[2] == data.K
    assert fit["estimate_K"]["Delta"].shape[2] == data.K

    print("Sanity checks passed.")


if __name__ == "__main__":
    main()