import numpy as np
import pytest

from src.simulation import LeniaConfig, LeniaSimulation


def make_config(**overrides):
    params = {
        "grid_size": 32,
        "kernel_radius": 5,
        "seed": 123,
        "init_type": "noise",
        "dt": 0.1,
    }
    params.update(overrides)
    return LeniaConfig(**params)


def test_kernel_is_normalized_and_finite():
    sim = LeniaSimulation(make_config())

    assert sim.kernel.shape == (11, 11)
    assert np.all(np.isfinite(sim.kernel))
    assert np.all(sim.kernel >= 0.0)
    assert np.isclose(sim.kernel.sum(), 1.0)


def test_reference_engine_is_deterministic_for_same_state():
    cfg = make_config()
    sim_a = LeniaSimulation(cfg)
    sim_b = LeniaSimulation(cfg)

    state = sim_a.get_state()
    sim_b.set_state(state)

    growth_a = sim_a.step()
    growth_b = sim_b.step()

    np.testing.assert_allclose(growth_a, growth_b, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(sim_a.world, sim_b.world, rtol=0.0, atol=1e-12)


def test_reference_engine_has_toroidal_translation_invariance():
    cfg = make_config(init_type="uniform_disk")
    sim = LeniaSimulation(cfg)
    shifted = LeniaSimulation(cfg)

    rng = np.random.default_rng(7)
    state = rng.uniform(0.0, 1.0, size=sim.world.shape)
    shift = (6, -4)

    sim.set_state(state)
    shifted.set_state(np.roll(state, shift, axis=(0, 1)))

    growth = sim.step()
    growth_shifted = shifted.step()

    np.testing.assert_allclose(
        growth_shifted,
        np.roll(growth, shift, axis=(0, 1)),
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        shifted.world,
        np.roll(sim.world, shift, axis=(0, 1)),
        rtol=1e-11,
        atol=1e-11,
    )


def test_invalid_kernel_geometry_is_rejected():
    with pytest.raises(ValueError, match="kernel diameter"):
        LeniaSimulation(make_config(grid_size=8, kernel_radius=5))
