# Tests for the population-block-major -> channel-major refactor of the W
# (connectivity) matrix and the parameters that share its population axis
# (H, D, S, Gamma). Covers:
#   - channel_major_perm() itself (pymindy/dataset/create_synth_data.py)
#   - the structural pattern it produces in create_synth_MINDy/HMMINDy's
#     output (pymindy/dataset/create_synth_data.py)
#   - that hmmindy_core.py's full W/mask and its Wee/Wei/Wie/Wii slices
#     agree with that same channel-major layout
#     (pymindy/training/hmmindy_core.py)
#
# Channel-major convention under test: index 2*c is the excitatory
# population of channel c, index 2*c + 1 is its inhibitory population.

import numpy as np
import pytest
import torch

from pymindy.dataset.create_synth_data import (
    channel_major_perm,
    create_synth_HMMINDy,
    create_synth_MINDy,
)
from pymindy.training.hmmindy_core import HMMINDyCore


# --------------------------------------------------------------------------
# channel_major_perm()
# --------------------------------------------------------------------------

def test_channel_major_perm_two_pops():
    # nChan=3, nPop=2: block-major order is [E0,E1,E2,I0,I1,I2];
    # channel-major order should be [E0,I0,E1,I1,E2,I2].
    perm = channel_major_perm(nChan=3, nPop=2)
    np.testing.assert_array_equal(perm, [0, 3, 1, 4, 2, 5])


def test_channel_major_perm_three_pops():
    # nChan=2, nPop=3: block-major [A0,A1,B0,B1,C0,C1] ->
    # channel-major [A0,B0,C0,A1,B1,C1].
    perm = channel_major_perm(nChan=2, nPop=3)
    np.testing.assert_array_equal(perm, [0, 2, 4, 1, 3, 5])


def test_channel_major_perm_reorders_vector():
    nChan, nPop = 5, 2
    perm = channel_major_perm(nChan, nPop)

    # Block-major vector where each entry encodes (population, channel).
    x_block = np.array([100 * p + c for p in range(nPop) for c in range(nChan)])
    x_chan = x_block[perm]

    expected = np.array([100 * p + c for c in range(nChan) for p in range(nPop)])
    np.testing.assert_array_equal(x_chan, expected)


def test_channel_major_perm_reorders_matrix():
    nChan, nPop = 4, 2
    perm = channel_major_perm(nChan, nPop)
    n = nChan * nPop
    M_block = np.arange(n * n).reshape(n, n)
    M_chan = M_block[np.ix_(perm, perm)]

    for c in range(nChan):
        for p in range(nPop):
            new_i = nPop * c + p
            old_i = p * nChan + c
            for c2 in range(nChan):
                for p2 in range(nPop):
                    new_j = nPop * c2 + p2
                    old_j = p2 * nChan + c2
                    assert M_chan[new_i, new_j] == M_block[old_i, old_j]


def test_channel_major_perm_labeled_matrix_before_after():
    # Visual before/after check: label every W entry by block and channels,
    # e.g. "Wei12" = I (row) <- E (col), row channel 1, col channel 2,
    # matching the Wee/Wei/Wie/Wii block naming in create_synth_data.py.
    # Run with `pytest -s` to see the printed matrices.

    # to see this actually printed, run with `pytest -s`
    # full command line input: `pytest -s tests/test_channel_major.py::test_channel_major_perm_labeled_matrix_before_after`
    nChan, nPop = 3, 2
    pops = "ei"
    labels = [(pops[p], c + 1) for p in range(nPop) for c in range(nChan)]
    M_block = np.array([[f"W{cp}{rp}{rc}{cc}" for cp, cc in labels]
                        for rp, rc in labels])

    perm = channel_major_perm(nChan, nPop)
    M_chan = M_block[np.ix_(perm, perm)]

    print("\nPopulation-block-major:")
    print(M_block)
    print("\nChannel-major:")
    print(M_chan)

    # Channel-major: index 2*c is E of channel c, 2*c + 1 is I of channel c.
    for i in range(nPop * nChan):
        for j in range(nPop * nChan):
            rp, rc = pops[i % nPop], i // nPop + 1
            cp, cc = pops[j % nPop], j // nPop + 1
            assert M_chan[i, j] == f"W{cp}{rp}{rc}{cc}"


# --------------------------------------------------------------------------
# create_synth_MINDy() / create_synth_HMMINDy(): channel-major structure
# --------------------------------------------------------------------------

@pytest.fixture
def synth_mindy_model():
    nChan, nPop = 10, 2
    model = create_synth_MINDy(nChan, nPop, pConn=25, tRun=50)
    return nChan, nPop, model


def test_create_synth_mindy_shapes(synth_mindy_model):
    nChan, nPop, model = synth_mindy_model
    n = nChan * nPop
    assert model["Params"]["W"].shape == (n, n)
    assert model["H"].shape == (nChan, n)
    assert model["Params"]["D"].shape == (n,)
    assert model["Params"]["S"].shape == (n,)


def test_w_local_only_blocks_are_diagonal_in_channel_major(synth_mindy_model):
    # In channel-major order, the E<-I and I<-I connections are purely local
    # (recurrent, same channel only) -- their off-diagonal entries must be
    # exactly zero, and their diagonal entries must be nonzero. Long-range
    # E<-E and I<-E connections are NOT restricted this way, so this is the
    # sharpest deterministic signature that the permutation landed correctly.
    nChan, nPop, model = synth_mindy_model
    W = model["Params"]["W"]
    E = np.arange(0, nPop * nChan, nPop)
    I = np.arange(1, nPop * nChan, nPop)

    W_ii = W[np.ix_(I, I)]
    W_ei_local = W[np.ix_(E, I)]

    assert np.allclose(W_ii - np.diag(np.diag(W_ii)), 0)
    assert np.allclose(W_ei_local - np.diag(np.diag(W_ei_local)), 0)
    assert np.all(np.diag(W_ii) != 0)
    assert np.all(np.diag(W_ei_local) != 0)


def test_h_has_no_inhibitory_contribution_in_channel_major(synth_mindy_model):
    # H is built with all-zero columns for the inhibitory population; after
    # permutation those must land exactly on the odd (inhibitory) columns.
    nChan, nPop, model = synth_mindy_model
    H = model["H"]
    E = np.arange(0, nPop * nChan, nPop)
    I = np.arange(1, nPop * nChan, nPop)

    assert np.allclose(H[:, I], 0)
    assert not np.allclose(H[:, E], 0)


def test_s_alternates_se_si_in_channel_major(synth_mindy_model):
    # S is deterministic (Se=2.5, Si=1.0 constants), so this exactly pins
    # down the E/I interleaving regardless of RNG.
    nChan, nPop, model = synth_mindy_model
    S = model["Params"]["S"]
    E = np.arange(0, nPop * nChan, nPop)
    I = np.arange(1, nPop * nChan, nPop)

    np.testing.assert_allclose(S[E], 2.5)
    np.testing.assert_allclose(S[I], 1.0)


def test_d_is_constant_within_each_population_block(synth_mindy_model):
    nChan, nPop, model = synth_mindy_model
    D = model["Params"]["D"]
    E = np.arange(0, nPop * nChan, nPop)
    I = np.arange(1, nPop * nChan, nPop)

    assert np.allclose(D[E], D[E][0])
    assert np.allclose(D[I], D[I][0])


def test_gamma_shares_w_zero_pattern_in_channel_major():
    # Gamma is generated as abs(sign(W)) * (outer product); it must respect
    # the exact same channel-major connectivity pattern as W.
    nChan, nPop, nS = 8, 2, 3
    model = create_synth_HMMINDy(nChan, nPop, nS, pConn=25, tRun=50)
    W = model["Params"]["W"]
    Gamma = model["Params"]["Gamma"]
    Wmask = np.sign(W)

    assert Gamma.shape == (nPop * nChan, nPop * nChan, nS)
    for k in range(nS):
        assert np.all(Gamma[Wmask == 0, k] == 0)


# --------------------------------------------------------------------------
# hmmindy_core.py: W and its E/I slices agree with create_synth_data.py's
# channel-major convention
# --------------------------------------------------------------------------

def test_hmmindy_core_init_w_mask_matches_channel_major_truth(synth_mindy_model):
    # HMMINDyCore.init() stores the full N x N mask; the E/I slices of the
    # initial W must respect the same channel-major pattern as the truth.
    nChan, nPop, model = synth_mindy_model
    W_mask = np.sign(model["Params"]["W"])
    E = np.arange(0, nPop * nChan, nPop)
    I = np.arange(1, nPop * nChan, nPop)

    H = np.zeros((nChan, nPop * nChan))
    Q = np.eye(nPop * nChan)
    R = np.eye(nChan)

    core = HMMINDyCore.init(H, Q, R, n_HMM=2, W_mask=W_mask, device="cpu")

    np.testing.assert_array_equal(core._W_mask.numpy(), W_mask != 0)
    W_fit = core.W.detach().numpy()
    assert np.all(W_fit[W_mask == 0] == 0)
    assert np.all(W_fit[:, E] >= 0) and np.all(W_fit[:, I] <= 0)
    np.testing.assert_array_equal(core.Wee.detach().numpy() != 0, W_fit[np.ix_(E, E)] != 0)
    np.testing.assert_array_equal(core.Wei.detach().numpy() != 0, W_fit[np.ix_(I, E)] != 0)


def test_hmmindy_core_w_property_roundtrips_channel_major_truth(synth_mindy_model):
    # Load a real channel-major W (from create_synth_data.py) into a core as
    # |W| magnitudes, and confirm the .W property reintroduces the E/I signs
    # and that the Wee/Wei/Wie/Wii slices match the old nPop=2 layout.
    nChan, nPop, model = synth_mindy_model
    W_true = model["Params"]["W"]
    E = np.arange(0, nPop * nChan, nPop)
    I = np.arange(1, nPop * nChan, nPop)

    H = np.zeros((nChan, nPop * nChan))
    Q = np.eye(nPop * nChan)
    R = np.eye(nChan)
    core = HMMINDyCore.init(H, Q, R, n_HMM=2, W_mask=np.sign(W_true), device="cpu")
    with torch.no_grad():
        core._W_raw.copy_(torch.from_numpy(np.abs(W_true)).float())

    np.testing.assert_allclose(core.W.detach().numpy(), W_true, atol=1e-5)
    np.testing.assert_allclose(core.Wee.detach().numpy(), W_true[np.ix_(E, E)], atol=1e-5)
    np.testing.assert_allclose(core.Wei.detach().numpy(), W_true[np.ix_(I, E)], atol=1e-5)
    np.testing.assert_allclose(core.Wie.detach().numpy(), -np.diag(W_true[np.ix_(E, I)]), atol=1e-5)
    np.testing.assert_allclose(core.Wii.detach().numpy(), -np.diag(W_true[np.ix_(I, I)]), atol=1e-5)


@pytest.mark.xfail(
    reason=(
        "Pre-existing bug unrelated to the channel-major refactor: "
        "HMMINDyCore.func_F() references self.n_pair, which is never "
        "defined on HMMINDyCore (only on the sibling pyMINDyCoreClassic)."
    ),
    strict=False,
)
def test_hmmindy_core_func_f_smoke():
    nChan, nPop, nS = 6, 2, 2
    model = create_synth_HMMINDy(nChan, nPop, nS, pConn=25, tRun=50)
    W_mask = np.sign(model["Params"]["W"])
    H = model["H"]
    Q = model["Q"]
    R = model["R"]

    torch.manual_seed(0)
    core = HMMINDyCore.init(H, Q, R, n_HMM=nS, W_mask=W_mask, device="cpu")

    x = torch.zeros(nPop * nChan)
    x_next = core.func_F(x, 0)
    assert x_next.shape == x.shape
